"""
Unit tests for RateLimiter implementation.
Tests token bucket algorithm, async compatibility, and rate limit configuration.
"""

import asyncio
import pytest
import time
from unittest.mock import patch

from src.data.rate_limiter import (
    RateLimiter,
    MultiRateLimiter,
    RateLimitConfig,
    rate_limited_call,
    create_rate_limiter_from_config
)


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.rate == 5
        assert config.period == 1
        assert config.burst_size == 5  # Should default to rate
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(rate=10, period=2, burst_size=15)
        
        assert config.rate == 10
        assert config.period == 2
        assert config.burst_size == 15
    
    def test_burst_size_defaults_to_rate(self):
        """Test that burst_size defaults to rate when not specified."""
        config = RateLimitConfig(rate=7, period=3)
        
        assert config.burst_size == 7


class TestRateLimiter:
    """Test RateLimiter functionality."""
    
    def test_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(rate=5, period=1, burst_size=10)
        
        assert limiter.rate == 5
        assert limiter.period == 1
        assert limiter.burst_size == 10
        assert limiter.tokens == 10.0
        assert limiter.get_available_tokens() == 10.0
    
    def test_default_burst_size(self):
        """Test that burst_size defaults to rate."""
        limiter = RateLimiter(rate=3, period=2)
        
        assert limiter.burst_size == 3
        assert limiter.tokens == 3.0
    
    @pytest.mark.asyncio
    async def test_immediate_token_acquisition(self):
        """Test acquiring tokens when available."""
        limiter = RateLimiter(rate=5, period=1)
        
        # Should acquire immediately
        wait_time = await limiter.acquire()
        
        assert wait_time == 0.0
        assert limiter.tokens == 4.0
        
        # Acquire multiple tokens
        wait_time = await limiter.acquire(3)
        
        assert wait_time == 0.0
        assert limiter.tokens == 1.0
    
    @pytest.mark.asyncio
    async def test_token_exhaustion_and_waiting(self):
        """Test behavior when tokens are exhausted."""
        limiter = RateLimiter(rate=2, period=1)  # 2 tokens per second
        
        # Exhaust all tokens
        await limiter.acquire(2)
        assert limiter.tokens == 0.0
        
        # Next acquisition should wait
        start_time = time.time()
        wait_time = await limiter.acquire()
        end_time = time.time()
        
        assert wait_time > 0
        assert end_time - start_time >= 0.4  # Should wait at least 0.4 seconds for 1 token
    
    @pytest.mark.asyncio
    async def test_try_acquire_success(self):
        """Test try_acquire when tokens are available."""
        limiter = RateLimiter(rate=5, period=1)
        
        # Should succeed
        success = await limiter.try_acquire()
        assert success is True
        assert limiter.tokens == 4.0
        
        # Should succeed with multiple tokens
        success = await limiter.try_acquire(3)
        assert success is True
        assert limiter.tokens == 1.0
    
    @pytest.mark.asyncio
    async def test_try_acquire_failure(self):
        """Test try_acquire when tokens are not available."""
        limiter = RateLimiter(rate=2, period=1)
        
        # Exhaust tokens
        await limiter.acquire(2)
        
        # Should fail
        success = await limiter.try_acquire()
        assert success is False
        assert limiter.tokens == 0.0
    
    @pytest.mark.asyncio
    async def test_token_refill_over_time(self):
        """Test that tokens are refilled over time."""
        limiter = RateLimiter(rate=10, period=1)  # 10 tokens per second
        
        # Exhaust tokens
        await limiter.acquire(10)
        assert limiter.tokens == 0.0
        
        # Wait for refill
        await asyncio.sleep(0.5)  # Wait 0.5 seconds
        
        # Should have ~5 tokens refilled
        available = limiter.get_available_tokens()
        assert 4.0 <= available <= 6.0  # Allow some tolerance
    
    def test_get_wait_time_estimation(self):
        """Test wait time estimation."""
        limiter = RateLimiter(rate=4, period=1)  # 4 tokens per second
        
        # With full tokens, no wait time
        wait_time = limiter.get_wait_time(1)
        assert wait_time == 0.0
        
        # Exhaust tokens
        limiter.tokens = 0.0
        
        # Should estimate 0.25 seconds for 1 token
        wait_time = limiter.get_wait_time(1)
        assert abs(wait_time - 0.25) < 0.01
        
        # Should estimate 0.5 seconds for 2 tokens
        wait_time = limiter.get_wait_time(2)
        assert abs(wait_time - 0.5) < 0.01
    
    @pytest.mark.asyncio
    async def test_excessive_token_request(self):
        """Test requesting more tokens than burst size."""
        limiter = RateLimiter(rate=5, period=1, burst_size=3)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="exceeds burst size"):
            await limiter.acquire(5)
    
    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """Test concurrent token acquisitions."""
        limiter = RateLimiter(rate=2, period=1)
        
        async def acquire_token():
            return await limiter.acquire()
        
        # Start multiple concurrent acquisitions
        tasks = [acquire_token() for _ in range(4)]
        wait_times = await asyncio.gather(*tasks)
        
        # First 2 should be immediate, next 2 should wait
        immediate_acquisitions = sum(1 for wt in wait_times if wt == 0.0)
        waited_acquisitions = sum(1 for wt in wait_times if wt > 0.0)
        
        assert immediate_acquisitions == 2
        assert waited_acquisitions == 2
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        limiter = RateLimiter(rate=5, period=1)
        
        # Initial stats
        stats = limiter.get_statistics()
        assert stats['requests_made'] == 0
        assert stats['tokens_consumed'] == 0
        assert stats['wait_time_total'] == 0.0
        assert stats['max_wait_time'] == 0.0
    
    @pytest.mark.asyncio
    async def test_statistics_updates(self):
        """Test that statistics are updated correctly."""
        limiter = RateLimiter(rate=5, period=1)
        
        # Make some requests
        await limiter.acquire(2)
        await limiter.acquire(1)
        
        stats = limiter.get_statistics()
        assert stats['requests_made'] == 2
        assert stats['tokens_consumed'] == 3
        assert stats['rate'] == 5
        assert stats['period'] == 1
        assert stats['burst_size'] == 5
    
    def test_reset_functionality(self):
        """Test reset functionality."""
        limiter = RateLimiter(rate=5, period=1)
        
        # Modify state
        limiter.tokens = 2.0
        limiter.stats['requests_made'] = 10
        
        # Reset
        limiter.reset()
        
        assert limiter.tokens == 5.0  # Back to burst_size
        assert limiter.stats['requests_made'] == 0


class TestMultiRateLimiter:
    """Test MultiRateLimiter functionality."""
    
    def test_initialization(self):
        """Test multi rate limiter initializes correctly."""
        multi_limiter = MultiRateLimiter()
        
        assert len(multi_limiter.limiters) == 0
    
    def test_add_limiter(self):
        """Test adding rate limiters."""
        multi_limiter = MultiRateLimiter()
        
        limiter = multi_limiter.add_limiter("api1", rate=5, period=1)
        
        assert isinstance(limiter, RateLimiter)
        assert "api1" in multi_limiter.limiters
        assert multi_limiter.limiters["api1"] == limiter
    
    def test_get_limiter(self):
        """Test getting rate limiters."""
        multi_limiter = MultiRateLimiter()
        
        # Add limiter
        original_limiter = multi_limiter.add_limiter("api1", rate=5, period=1)
        
        # Get limiter
        retrieved_limiter = multi_limiter.get_limiter("api1")
        assert retrieved_limiter == original_limiter
        
        # Get non-existent limiter
        assert multi_limiter.get_limiter("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_acquire_from_specific_limiter(self):
        """Test acquiring tokens from specific limiter."""
        multi_limiter = MultiRateLimiter()
        
        multi_limiter.add_limiter("api1", rate=5, period=1)
        multi_limiter.add_limiter("api2", rate=10, period=1)
        
        # Acquire from api1
        wait_time = await multi_limiter.acquire("api1", 2)
        assert wait_time == 0.0
        
        # Check that api1 tokens were consumed
        api1_limiter = multi_limiter.get_limiter("api1")
        assert api1_limiter.tokens == 3.0
        
        # Check that api2 tokens were not affected
        api2_limiter = multi_limiter.get_limiter("api2")
        assert api2_limiter.tokens == 10.0
    
    @pytest.mark.asyncio
    async def test_acquire_nonexistent_limiter(self):
        """Test acquiring from non-existent limiter."""
        multi_limiter = MultiRateLimiter()
        
        with pytest.raises(KeyError, match="not found"):
            await multi_limiter.acquire("nonexistent")
    
    @pytest.mark.asyncio
    async def test_try_acquire_from_specific_limiter(self):
        """Test try_acquire from specific limiter."""
        multi_limiter = MultiRateLimiter()
        
        multi_limiter.add_limiter("api1", rate=2, period=1)
        
        # Should succeed
        success = await multi_limiter.try_acquire("api1")
        assert success is True
        
        # Exhaust tokens
        await multi_limiter.acquire("api1")
        
        # Should fail
        success = await multi_limiter.try_acquire("api1")
        assert success is False
    
    def test_get_all_statistics(self):
        """Test getting statistics for all limiters."""
        multi_limiter = MultiRateLimiter()
        
        multi_limiter.add_limiter("api1", rate=5, period=1)
        multi_limiter.add_limiter("api2", rate=10, period=2)
        
        stats = multi_limiter.get_statistics()
        
        assert "api1" in stats
        assert "api2" in stats
        assert stats["api1"]["rate"] == 5
        assert stats["api2"]["rate"] == 10
    
    def test_reset_all_limiters(self):
        """Test resetting all limiters."""
        multi_limiter = MultiRateLimiter()
        
        limiter1 = multi_limiter.add_limiter("api1", rate=5, period=1)
        limiter2 = multi_limiter.add_limiter("api2", rate=10, period=1)
        
        # Modify state
        limiter1.tokens = 2.0
        limiter2.tokens = 5.0
        
        # Reset all
        multi_limiter.reset_all()
        
        assert limiter1.tokens == 5.0
        assert limiter2.tokens == 10.0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_rate_limited_call_async(self):
        """Test rate limited call with async function."""
        limiter = RateLimiter(rate=5, period=1)
        
        async def async_func(x, y):
            return x + y
        
        result = await rate_limited_call(async_func, limiter, 3, 4)
        
        assert result == 7
        assert limiter.tokens == 4.0  # One token consumed
    
    @pytest.mark.asyncio
    async def test_rate_limited_call_sync(self):
        """Test rate limited call with sync function."""
        limiter = RateLimiter(rate=5, period=1)
        
        def sync_func(x, y):
            return x * y
        
        result = await rate_limited_call(sync_func, limiter, 3, 4)
        
        assert result == 12
        assert limiter.tokens == 4.0  # One token consumed
    
    def test_create_rate_limiter_from_config(self):
        """Test creating rate limiter from configuration."""
        config = RateLimitConfig(rate=10, period=2, burst_size=15)
        
        limiter = create_rate_limiter_from_config(config)
        
        assert limiter.rate == 10
        assert limiter.period == 2
        assert limiter.burst_size == 15
        assert limiter.tokens == 15.0


class TestRateLimiterIntegration:
    """Integration tests for rate limiter."""
    
    @pytest.mark.asyncio
    async def test_realistic_api_rate_limiting(self):
        """Test realistic API rate limiting scenario."""
        # Simulate Yahoo Finance rate limit: 5 requests per second
        limiter = RateLimiter(rate=5, period=1)
        
        async def mock_api_call(ticker):
            """Mock API call that takes some time."""
            await asyncio.sleep(0.01)  # Simulate network delay
            return f"data_for_{ticker}"
        
        # Make 10 API calls
        tickers = [f"STOCK{i}" for i in range(10)]
        
        start_time = time.time()
        results = []
        
        for ticker in tickers:
            await limiter.acquire()
            result = await mock_api_call(ticker)
            results.append(result)
        
        end_time = time.time()
        
        # Should take at least 1 second due to rate limiting
        # (first 5 immediate, next 5 after 1 second)
        assert end_time - start_time >= 0.8  # Allow some tolerance
        assert len(results) == 10
        
        # Check statistics
        stats = limiter.get_statistics()
        assert stats['requests_made'] == 10
        assert stats['tokens_consumed'] == 10
    
    @pytest.mark.asyncio
    async def test_burst_handling(self):
        """Test burst request handling."""
        # Allow burst of 10 requests, but only 2 per second sustained
        limiter = RateLimiter(rate=2, period=1, burst_size=10)
        
        # Make burst of 10 requests
        start_time = time.time()
        
        for _ in range(10):
            await limiter.acquire()
        
        burst_time = time.time() - start_time
        
        # Burst should be handled quickly
        assert burst_time < 0.5
        
        # But next request should wait
        start_time = time.time()
        await limiter.acquire()
        wait_time = time.time() - start_time
        
        # Should wait for token refill
        assert wait_time >= 0.4  # At least 0.4 seconds for next token
    
    @pytest.mark.asyncio
    async def test_multiple_services_isolation(self):
        """Test that multiple services don't interfere with each other."""
        multi_limiter = MultiRateLimiter()
        
        # Different rate limits for different services
        multi_limiter.add_limiter("yahoo", rate=5, period=1)
        multi_limiter.add_limiter("alpha_vantage", rate=1, period=1)
        
        # Exhaust yahoo tokens
        for _ in range(5):
            await multi_limiter.acquire("yahoo")
        
        # Alpha Vantage should still work
        success = await multi_limiter.try_acquire("alpha_vantage")
        assert success is True
        
        # Yahoo should be exhausted
        success = await multi_limiter.try_acquire("yahoo")
        assert success is False