import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse

from .schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class BaseAPIException(Exception):
    """Base exception class for API errors"""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.request_id = request_id or str(uuid.uuid4())
        super().__init__(self.message)


class ValidationError(BaseAPIException):
    """Input validation errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
            request_id=request_id,
        )


class AuthenticationError(BaseAPIException):
    """Authentication failures"""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
            request_id=request_id,
        )


class AuthorizationError(BaseAPIException):
    """Authorization failures"""

    def __init__(
        self,
        message: str = "Access denied",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details,
            request_id=request_id,
        )


class RateLimitError(BaseAPIException):
    """Rate limiting errors"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        details = {"retry_after_seconds": retry_after} if retry_after else None
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            details=details,
            request_id=request_id,
        )


class ModelError(BaseAPIException):
    """Model-related errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            status_code=422,
            details=details,
            request_id=request_id,
        )


class InferenceError(BaseAPIException):
    """Inference pipeline errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="INFERENCE_ERROR",
            status_code=500,
            details=details,
            request_id=request_id,
        )


class ServiceUnavailableError(BaseAPIException):
    """Service unavailability errors"""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=details,
            request_id=request_id,
        )


class TimeoutError(BaseAPIException):
    """Request timeout errors"""

    def __init__(
        self,
        message: str = "Request timeout",
        timeout_seconds: Optional[float] = None,
        request_id: Optional[str] = None,
    ):
        details = {"timeout_seconds": timeout_seconds} if timeout_seconds else None
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            status_code=408,
            details=details,
            request_id=request_id,
        )


class CacheError(BaseAPIException):
    """Cache-related errors (non-critical)"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details=details,
            request_id=request_id,
        )


async def api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """Handle custom API exceptions"""

    # Log the error
    logger.error(
        f"API Error [{exc.error_code}]: {exc.message}",
        extra={
            "request_id": exc.request_id,
            "status_code": exc.status_code,
            "error_code": exc.error_code,
            "details": exc.details,
            "path": str(request.url),
            "method": request.method,
        },
    )

    # Create error response
    error_detail = ErrorDetail(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        request_id=exc.request_id,
    )

    error_response = ErrorResponse(error=error_detail, timestamp=datetime.now())

    # Add retry-after header for rate limit errors
    headers = {}
    if isinstance(exc, RateLimitError) and exc.details.get("retry_after_seconds"):
        headers["Retry-After"] = str(exc.details["retry_after_seconds"])

    return JSONResponse(status_code=exc.status_code, content=error_response.dict(), headers=headers)


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic validation exceptions"""

    request_id = str(uuid.uuid4())

    # Extract validation details
    if hasattr(exc, "errors"):
        validation_errors = []
        for error in exc.errors():
            validation_errors.append(
                {
                    "field": " -> ".join(str(x) for x in error.get("loc", [])),
                    "message": error.get("msg", "Validation error"),
                    "type": error.get("type", "unknown"),
                    "input": error.get("input"),
                }
            )
        details = {"validation_errors": validation_errors}
    else:
        details = {"raw_error": str(exc)}

    logger.warning(
        f"Validation error: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": str(request.url),
            "method": request.method,
            "details": details,
        },
    )

    error_detail = ErrorDetail(
        error_code="VALIDATION_ERROR",
        message="Input validation failed",
        details=details,
        request_id=request_id,
    )

    error_response = ErrorResponse(error=error_detail, timestamp=datetime.now())

    return JSONResponse(status_code=422, content=error_response.dict())


async def http_exception_handler_custom(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions"""

    request_id = str(uuid.uuid4())

    # Map HTTP status codes to error codes
    status_to_error_code = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        410: "GONE",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_SERVER_ERROR",
        501: "NOT_IMPLEMENTED",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
        504: "GATEWAY_TIMEOUT",
    }

    error_code = status_to_error_code.get(exc.status_code, "HTTP_ERROR")

    logger.warning(
        f"HTTP Exception [{error_code}]: {exc.detail}",
        extra={
            "request_id": request_id,
            "status_code": exc.status_code,
            "path": str(request.url),
            "method": request.method,
        },
    )

    error_detail = ErrorDetail(
        error_code=error_code, message=str(exc.detail), request_id=request_id
    )

    error_response = ErrorResponse(error=error_detail, timestamp=datetime.now())

    return JSONResponse(status_code=exc.status_code, content=error_response.dict())


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions"""

    request_id = str(uuid.uuid4())

    # Log the full exception with traceback
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": str(request.url),
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
        exc_info=True,
    )

    # Don't expose internal error details in production
    error_detail = ErrorDetail(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        request_id=request_id,
    )

    error_response = ErrorResponse(error=error_detail, timestamp=datetime.now())

    return JSONResponse(status_code=500, content=error_response.dict())


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """Helper function to create standardized error responses"""

    request_id = request_id or str(uuid.uuid4())

    error_detail = ErrorDetail(
        error_code=error_code, message=message, details=details, request_id=request_id
    )

    error_response = ErrorResponse(error=error_detail, timestamp=datetime.now())

    return JSONResponse(status_code=status_code, content=error_response.dict())
