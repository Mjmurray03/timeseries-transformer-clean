"""
Rate limiter implementation using token bucket algorithm.
Provides async-compatible rate limiting for API requests.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    rate: int = 5  # requests per period
    period: int = 1  # period in seconds
    burst_size: Optional[int] = None  # max burst size (defaults to rate)
    
    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.rate


class RateLimiter:
    """
    Token bucket rate limiter with async support.
    
    Implements the token bucket algorithm to control the rate of requests.
    Supports burst requests up to the bucket capacity while maintaining
    the average rate over time.
    """
    
    def __init__(self, rate: int = 5, period: int = 1, burst_size: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of requests allowed per period
            period: Time period in seconds
            burst_size: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.period = period
        self.burst_size = burst_size or rate
        
        # Token bucket state
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        
        # Async lock for thread safety
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'requests_made': 0,
            'tokens_consumed': 0,
            'wait_time_total': 0.0,
            'max_wait_time': 0.0
        }
        
        logger.debug(f"RateLimiter initialized: {rate}/{period}s, burst={self.burst_size}")    

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens for request, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire (default 1)
            
        Returns:
            Time waited in seconds
        """
        if tokens > self.burst_size:
            raise ValueError(f"Requested tokens ({tokens}) exceeds burst size ({self.burst_size})")
        
        async with self._lock:
            wait_time = await self._acquire_tokens(tokens)
            
            # Update statistics
            self.stats['requests_made'] += 1
            self.stats['tokens_consumed'] += tokens
            self.stats['wait_time_total'] += wait_time
            self.stats['max_wait_time'] = max(self.stats['max_wait_time'], wait_time)
            
            return wait_time
    
    async def _acquire_tokens(self, tokens: int) -> float:
        """
        Internal method to acquire tokens with waiting.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
        """
        start_time = time.time()
        
        while True:
            # Refill tokens based on elapsed time
            await self._refill_tokens()
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                wait_time = time.time() - start_time
                
                if wait_time > 0:
                    logger.debug(f"Acquired {tokens} tokens after waiting {wait_time:.3f}s")
                else:
                    logger.debug(f"Acquired {tokens} tokens immediately")
                
                return wait_time
            
            # Calculate how long to wait for next token
            tokens_needed = tokens - self.tokens
            wait_duration = (tokens_needed / self.rate) * self.period
            
            # Wait for a fraction of the required time to avoid oversleeping
            sleep_time = min(wait_duration / 2, 0.1)
            await asyncio.sleep(sleep_time)
    
    async def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        if elapsed > 0:
            # Calculate tokens to add based on rate and elapsed time
            tokens_to_add = (elapsed / self.period) * self.rate
            
            # Add tokens but don't exceed burst size
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_update = now
            
            logger.debug(f"Refilled tokens: {self.tokens:.2f}/{self.burst_size}")
    
    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        if tokens > self.burst_size:
            return False
        
        async with self._lock:
            await self._refill_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.stats['requests_made'] += 1
                self.stats['tokens_consumed'] += tokens
                logger.debug(f"Acquired {tokens} tokens without waiting")
                return True
            
            logger.debug(f"Could not acquire {tokens} tokens (available: {self.tokens:.2f})")
            return False    
 
    def get_available_tokens(self) -> float:
        """
        Get number of currently available tokens (non-async).
        
        Returns:
            Number of available tokens
        """
        # Update tokens based on elapsed time
        now = time.time()
        elapsed = now - self.last_update
        
        if elapsed > 0:
            tokens_to_add = (elapsed / self.period) * self.rate
            available = min(self.burst_size, self.tokens + tokens_to_add)
        else:
            available = self.tokens
        
        return available
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Estimate wait time for acquiring tokens (non-async).
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Estimated wait time in seconds
        """
        available = self.get_available_tokens()
        
        if available >= tokens:
            return 0.0
        
        tokens_needed = tokens - available
        return (tokens_needed / self.rate) * self.period
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        stats = self.stats.copy()
        stats.update({
            'rate': self.rate,
            'period': self.period,
            'burst_size': self.burst_size,
            'available_tokens': self.get_available_tokens(),
            'avg_wait_time': (
                stats['wait_time_total'] / stats['requests_made'] 
                if stats['requests_made'] > 0 else 0.0
            )
        })
        return stats
    
    def reset_statistics(self):
        """Reset rate limiter statistics."""
        self.stats = {
            'requests_made': 0,
            'tokens_consumed': 0,
            'wait_time_total': 0.0,
            'max_wait_time': 0.0
        }
        logger.debug("Rate limiter statistics reset")
    
    def reset(self):
        """Reset rate limiter to initial state."""
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        self.reset_statistics()
        logger.debug("Rate limiter reset to initial state")


class MultiRateLimiter:
    """
    Manages multiple rate limiters for different services/endpoints.
    
    Useful for managing different rate limits for different APIs
    or endpoints within the same application.
    """
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        logger.debug("MultiRateLimiter initialized")
    
    def add_limiter(self, name: str, rate: int, period: int = 1, 
                   burst_size: Optional[int] = None) -> RateLimiter:
        """
        Add a rate limiter for a specific service.
        
        Args:
            name: Service/endpoint name
            rate: Requests per period
            period: Time period in seconds
            burst_size: Maximum burst size
            
        Returns:
            The created RateLimiter instance
        """
        limiter = RateLimiter(rate, period, burst_size)
        self.limiters[name] = limiter
        logger.info(f"Added rate limiter '{name}': {rate}/{period}s")
        return limiter    
   
    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """Get rate limiter by name."""
        return self.limiters.get(name)
    
    async def acquire(self, name: str, tokens: int = 1) -> float:
        """
        Acquire tokens from a specific rate limiter.
        
        Args:
            name: Service/endpoint name
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
            
        Raises:
            KeyError: If rate limiter not found
        """
        if name not in self.limiters:
            raise KeyError(f"Rate limiter '{name}' not found")
        
        return await self.limiters[name].acquire(tokens)
    
    async def try_acquire(self, name: str, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Args:
            name: Service/endpoint name
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
            
        Raises:
            KeyError: If rate limiter not found
        """
        if name not in self.limiters:
            raise KeyError(f"Rate limiter '{name}' not found")
        
        return await self.limiters[name].try_acquire(tokens)
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all rate limiters."""
        return {
            name: limiter.get_statistics() 
            for name, limiter in self.limiters.items()
        }
    
    def reset_all(self):
        """Reset all rate limiters."""
        for limiter in self.limiters.values():
            limiter.reset()
        logger.info("All rate limiters reset")


# Convenience functions for common use cases
async def rate_limited_call(func, limiter: RateLimiter, *args, **kwargs):
    """
    Execute a function with rate limiting.
    
    Args:
        func: Function to call
        limiter: RateLimiter instance
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    await limiter.acquire()
    return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)


def create_rate_limiter_from_config(config: RateLimitConfig) -> RateLimiter:
    """
    Create rate limiter from configuration.
    
    Args:
        config: RateLimitConfig instance
        
    Returns:
        Configured RateLimiter
    """
    return RateLimiter(
        rate=config.rate,
        period=config.period,
        burst_size=config.burst_size
    )