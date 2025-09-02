import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

import redis
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .exceptions import AuthenticationError, RateLimitError, create_error_response
from .schemas import AuthResponse, RateLimitInfo

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = str(hash(f"{request.url}{start_time}"))[:8]

        # Add request ID to request state
        request.state.request_id = request_id

        # Log incoming request
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
                "client_host": request.client.host if request.client else None,
            },
        )

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time * 1000, 2),
                },
            )

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time_ms": round(process_time * 1000, 2),
                },
                exc_info=True,
            )
            raise


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with specific configurations"""

    def __init__(
        self,
        app,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        max_age: int = 600,
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.max_age = max_age

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "OPTIONS":
            # Handle preflight requests
            response = Response()
            self._add_cors_headers(response, request)
            return response

        response = await call_next(request)
        self._add_cors_headers(response, request)
        return response

    def _add_cors_headers(self, response: Response, request: Request):
        origin = request.headers.get("origin")

        if origin and (origin in self.allow_origins or "*" in self.allow_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif "*" in self.allow_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"

        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
        response.headers["Access-Control-Allow-Credentials"] = "true"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis"""

    def __init__(
        self,
        app,
        redis_client: Optional[redis.Redis] = None,
        default_rate_limit: int = 100,
        time_window: int = 60,
        enable_per_user_limits: bool = True,
    ):
        super().__init__(app)
        self.redis_client = redis_client
        self.default_rate_limit = default_rate_limit
        self.time_window = time_window
        self.enable_per_user_limits = enable_per_user_limits
        self.memory_store = {}  # Fallback if Redis unavailable

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        rate_limit_info = await self._check_rate_limit(client_id, request)

        if rate_limit_info.remaining <= 0:
            # Rate limit exceeded
            retry_after = int((rate_limit_info.reset_at - datetime.now()).total_seconds())

            logger.warning(
                f"Rate limit exceeded for client: {client_id[:8]}...",
                extra={
                    "client_id": client_id[:8],
                    "limit": rate_limit_info.limit,
                    "retry_after": retry_after,
                },
            )

            return create_error_response(
                error_code="RATE_LIMIT_EXCEEDED",
                message="Rate limit exceeded",
                status_code=429,
                details={"retry_after_seconds": retry_after},
            )

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info.reset_at.timestamp()))

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get API key from Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        return f"ip:{client_ip}"

    async def _check_rate_limit(self, client_id: str, request: Request) -> RateLimitInfo:
        """Check and update rate limit for client"""
        now = datetime.now()
        window_start = now.replace(second=0, microsecond=0)
        window_end = window_start + timedelta(minutes=1)

        # Get user-specific limit if available
        rate_limit = await self._get_user_rate_limit(client_id, request)

        if self.redis_client:
            # Use Redis for distributed rate limiting
            try:
                key = f"rate_limit:{client_id}:{window_start.isoformat()}"

                # Increment counter
                current_requests = self.redis_client.incr(key)

                # Set expiration on first increment
                if current_requests == 1:
                    self.redis_client.expire(key, self.time_window)

                remaining = max(0, rate_limit - current_requests)

                return RateLimitInfo(limit=rate_limit, remaining=remaining, reset_at=window_end)

            except redis.RedisError as e:
                logger.error(f"Redis error in rate limiting: {e}")
                # Fall through to memory-based limiting

        # Memory-based rate limiting (fallback)
        key = f"{client_id}:{window_start.isoformat()}"

        if key not in self.memory_store:
            self.memory_store[key] = {"count": 0, "expires": window_end}

        # Clean up expired entries
        expired_keys = [k for k, v in self.memory_store.items() if v["expires"] < now]
        for k in expired_keys:
            del self.memory_store[k]

        # Increment counter
        self.memory_store[key]["count"] += 1
        current_requests = self.memory_store[key]["count"]

        remaining = max(0, rate_limit - current_requests)

        return RateLimitInfo(limit=rate_limit, remaining=remaining, reset_at=window_end)

    async def _get_user_rate_limit(self, client_id: str, request: Request) -> int:
        """Get rate limit for specific user/API key"""
        # TODO: Implement user-specific rate limits from database
        # For now, return default rate limit
        return self.default_rate_limit


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Content-Security-Policy"] = csp_policy

        return response


# Authentication classes
class APIKeyAuth:
    """API Key authentication handler"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.api_keys = {}  # In-memory store for development

    async def authenticate(self, api_key: str) -> AuthResponse:
        """Authenticate API key"""
        if not api_key:
            raise AuthenticationError("API key required")

        # Hash the API key for secure storage lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Check Redis cache first
        if self.redis_client:
            try:
                cached_auth = self.redis_client.get(f"auth:{key_hash}")
                if cached_auth:
                    import json

                    auth_data = json.loads(cached_auth)
                    return AuthResponse(**auth_data)
            except redis.RedisError:
                pass

        # Validate API key (placeholder implementation)
        if await self._validate_api_key(api_key):
            auth_response = AuthResponse(
                authenticated=True,
                user_id=key_hash[:16],
                rate_limit=100,
                expires_at=datetime.now() + timedelta(hours=1),
            )

            # Cache authentication result
            if self.redis_client:
                try:
                    import json

                    self.redis_client.setex(
                        f"auth:{key_hash}",
                        3600,  # 1 hour cache
                        json.dumps(auth_response.dict(), default=str),
                    )
                except redis.RedisError:
                    pass

            return auth_response

        raise AuthenticationError("Invalid API key")

    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key against store"""
        # TODO: Implement actual API key validation against database
        # For development, accept any key that's at least 32 characters
        return len(api_key) >= 32


# Dependency functions
security = HTTPBearer(auto_error=False)
api_key_auth = APIKeyAuth()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> AuthResponse:
    """Dependency to get current authenticated user"""
    if not credentials:
        raise AuthenticationError("Authentication required")

    return await api_key_auth.authenticate(credentials.credentials)


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[AuthResponse]:
    """Dependency to get current user (optional)"""
    if not credentials:
        return None

    try:
        return await api_key_auth.authenticate(credentials.credentials)
    except AuthenticationError:
        return None


def require_rate_limit_check(limit: Optional[int] = None):
    """Dependency factory for custom rate limits"""

    async def _check_rate_limit(
        request: Request, user: Optional[AuthResponse] = Depends(get_optional_user)
    ):
        # Rate limiting logic would go here
        # This is handled by the middleware, so this is mainly for custom limits
        pass

    return _check_rate_limit
