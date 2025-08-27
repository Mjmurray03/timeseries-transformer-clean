"""
Unit tests for Redis connection management.

Tests connection pooling, health checks, retry logic, and graceful shutdown.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

import redis
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from src.cache.config import RedisConfig, create_test_config
from src.cache.connection import RedisConnectionManager, with_redis_retry
from src.cache.exceptions import RedisConnectionError, CacheOperationError


class TestRedisConnectionManager:
    """Test suite for RedisConnectionManager"""
    
    @pytest.fixture
    def config(self):
        """Create test Redis configuration"""
        return create_test_config()
    
    @pytest.fixture
    def connection_manager(self, config):
        """Create RedisConnectionManager instance for testing"""
        return RedisConnectionManager(config)
    
    def test_initialization(self, config):
        """Test RedisConnectionManager initializes correctly"""
        manager = RedisConnectionManager(config)
        
        assert manager.config == config
        assert manager._pools == {}
        assert manager._async_pools == {}
        assert manager._clients == {}
        assert manager._async_clients == {}
        assert manager._is_healthy is True
        assert manager._shutdown is False
    
    @patch('redis.Redis')
    def test_get_connection_success(self, mock_redis, connection_manager):
        """Test successful Redis connection creation"""
        # Setup mock
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        # Test connection
        client = connection_manager.get_connection(0)
        
        assert client == mock_client
        assert 0 in connection_manager._clients
        mock_client.ping.assert_called_once()
    
    @patch('redis.Redis')
    def test_get_connection_failure(self, mock_redis, connection_manager):
        """Test Redis connection failure handling"""
        # Setup mock to raise connection error
        mock_redis.side_effect = ConnectionError("Connection failed")
        
        # Test connection failure
        with pytest.raises(RedisConnectionError) as exc_info:
            connection_manager.get_connection(0)
        
        assert "Failed to connect to Redis database 0" in str(exc_info.value)
        assert 0 not in connection_manager._clients
    
    @patch('redis.Redis')
    def test_get_connection_reuse(self, mock_redis, connection_manager):
        """Test connection reuse for same database"""
        # Setup mock
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        # Get connection twice
        client1 = connection_manager.get_connection(0)
        client2 = connection_manager.get_connection(0)
        
        # Should return same client
        assert client1 == client2
        assert mock_redis.call_count == 1  # Only created once
    
    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    async def test_get_async_connection_success(self, mock_redis, connection_manager):
        """Test successful async Redis connection creation"""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis.return_value = mock_client
        
        # Test async connection
        client = await connection_manager.get_async_connection(0)
        
        assert client == mock_client
        assert 0 in connection_manager._async_clients
        mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    async def test_get_async_connection_failure(self, mock_redis, connection_manager):
        """Test async Redis connection failure handling"""
        # Setup mock to raise connection error
        mock_redis.side_effect = ConnectionError("Async connection failed")
        
        # Test connection failure
        with pytest.raises(RedisConnectionError) as exc_info:
            await connection_manager.get_async_connection(0)
        
        assert "Failed to connect to async Redis database 0" in str(exc_info.value)
        assert 0 not in connection_manager._async_clients
    
    @pytest.mark.asyncio
    async def test_get_connection_context(self, connection_manager):
        """Test async connection context manager"""
        mock_client = AsyncMock()
        
        with patch.object(connection_manager, 'get_async_connection', return_value=mock_client):
            async with connection_manager.get_connection_context(0) as client:
                assert client == mock_client
    
    @patch('redis.Redis')
    def test_health_check_success(self, mock_redis, connection_manager):
        """Test successful health check"""
        # Setup mock
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.setex.return_value = True
        mock_client.get.return_value = "test_value"
        mock_client.delete.return_value = 1
        mock_redis.return_value = mock_client
        
        # Test health check
        result = connection_manager.health_check()
        
        assert result is True
        assert connection_manager._is_healthy is True
        mock_client.ping.assert_called_once()
        mock_client.setex.assert_called_once()
        mock_client.get.assert_called_once()
        mock_client.delete.assert_called_once()
    
    @patch('redis.Redis')
    def test_health_check_failure(self, mock_redis, connection_manager):
        """Test health check failure"""
        # Setup mock to fail ping
        mock_client = Mock()
        mock_client.ping.side_effect = ConnectionError("Ping failed")
        mock_redis.return_value = mock_client
        
        # Test health check
        result = connection_manager.health_check()
        
        assert result is False
        assert connection_manager._is_healthy is False
    
    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    async def test_async_health_check_success(self, mock_redis, connection_manager):
        """Test successful async health check"""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.setex.return_value = True
        mock_client.get.return_value = "async_test_value"
        mock_client.delete.return_value = 1
        mock_redis.return_value = mock_client
        
        # Test async health check
        result = await connection_manager.async_health_check()
        
        assert result is True
        assert connection_manager._is_healthy is True
        mock_client.ping.assert_called_once()
        mock_client.setex.assert_called_once()
        mock_client.get.assert_called_once()
        mock_client.delete.assert_called_once()
    
    def test_is_healthy_recent_check(self, connection_manager):
        """Test is_healthy with recent health check"""
        # Set recent health check
        connection_manager._is_healthy = True
        connection_manager._last_health_check = datetime.now().timestamp()
        
        result = connection_manager.is_healthy()
        assert result is True
    
    @patch.object(RedisConnectionManager, 'health_check')
    def test_is_healthy_stale_check(self, mock_health_check, connection_manager):
        """Test is_healthy with stale health check"""
        # Set old health check
        connection_manager._last_health_check = 0
        mock_health_check.return_value = True
        
        result = connection_manager.is_healthy()
        
        assert result is True
        mock_health_check.assert_called_once()
    
    def test_get_connection_info(self, connection_manager):
        """Test connection info retrieval"""
        # Add some mock connections
        connection_manager._clients[0] = Mock()
        connection_manager._async_clients[1] = Mock()
        connection_manager._is_healthy = True
        connection_manager._last_health_check = datetime.now().timestamp()
        
        info = connection_manager.get_connection_info()
        
        assert info["host"] == connection_manager.config.host
        assert info["port"] == connection_manager.config.port
        assert info["is_healthy"] is True
        assert info["active_pools"] == 0  # No pools created yet
        assert info["databases"] == [0]
        assert info["async_databases"] == [1]
    
    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    async def test_get_redis_info(self, mock_redis, connection_manager):
        """Test Redis server info retrieval"""
        # Setup mock
        mock_client = AsyncMock()
        mock_info = {
            "redis_version": "7.0.0",
            "os": "Linux",
            "arch_bits": 64
        }
        mock_client.info.return_value = mock_info
        mock_redis.return_value = mock_client
        
        # Test info retrieval
        info = await connection_manager.get_redis_info("server")
        
        assert info == mock_info
        mock_client.info.assert_called_once_with("server")
    
    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    async def test_get_redis_info_failure(self, mock_redis, connection_manager):
        """Test Redis info retrieval failure"""
        # Setup mock to fail
        mock_client = AsyncMock()
        mock_client.info.side_effect = RedisError("Info failed")
        mock_redis.return_value = mock_client
        
        # Test info retrieval failure
        with pytest.raises(CacheOperationError):
            await connection_manager.get_redis_info("server")
    
    def test_close_connections(self, connection_manager):
        """Test connection cleanup on close"""
        # Add mock connections
        mock_client = Mock()
        mock_pool = Mock()
        
        connection_manager._clients[0] = mock_client
        connection_manager._pools[0] = mock_pool
        
        # Test close
        connection_manager.close()
        
        assert connection_manager._shutdown is True
        mock_client.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert len(connection_manager._clients) == 0
        assert len(connection_manager._pools) == 0
    
    @pytest.mark.asyncio
    async def test_aclose_connections(self, connection_manager):
        """Test async connection cleanup"""
        # Add mock async connections
        mock_client = AsyncMock()
        mock_pool = AsyncMock()
        
        connection_manager._async_clients[0] = mock_client
        connection_manager._async_pools[0] = mock_pool
        
        # Test async close
        await connection_manager.aclose()
        
        assert connection_manager._shutdown is True
        mock_client.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert len(connection_manager._async_clients) == 0
        assert len(connection_manager._async_pools) == 0
    
    def test_context_manager(self, connection_manager):
        """Test synchronous context manager"""
        with patch.object(connection_manager, 'close') as mock_close:
            with connection_manager:
                pass
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, connection_manager):
        """Test asynchronous context manager"""
        with patch.object(connection_manager, 'aclose') as mock_aclose:
            async with connection_manager:
                pass
            mock_aclose.assert_called_once()
    
    def test_shutdown_prevents_new_connections(self, connection_manager):
        """Test that shutdown prevents new connections"""
        connection_manager._shutdown = True
        
        with pytest.raises(RedisConnectionError) as exc_info:
            connection_manager.get_connection(0)
        
        assert "Connection manager is shut down" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_shutdown_prevents_new_async_connections(self, connection_manager):
        """Test that shutdown prevents new async connections"""
        connection_manager._shutdown = True
        
        with pytest.raises(RedisConnectionError) as exc_info:
            await connection_manager.get_async_connection(0)
        
        assert "Connection manager is shut down" in str(exc_info.value)


class TestRedisRetryLogic:
    """Test suite for Redis retry logic"""
    
    @pytest.mark.asyncio
    async def test_with_redis_retry_success(self):
        """Test successful operation without retry"""
        async def successful_operation():
            return "success"
        
        result = await with_redis_retry(successful_operation)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_with_redis_retry_eventual_success(self):
        """Test operation that succeeds after retries"""
        call_count = 0
        
        async def eventually_successful_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await with_redis_retry(eventually_successful_operation, max_retries=3)
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_with_redis_retry_max_retries_exceeded(self):
        """Test operation that fails after max retries"""
        async def always_failing_operation():
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(CacheOperationError) as exc_info:
            await with_redis_retry(always_failing_operation, max_retries=2)
        
        assert "Redis operation failed after 3 attempts" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_with_redis_retry_non_redis_error(self):
        """Test that non-Redis errors are not retried"""
        async def operation_with_non_redis_error():
            raise ValueError("Not a Redis error")
        
        with pytest.raises(ValueError):
            await with_redis_retry(operation_with_non_redis_error)
    
    @pytest.mark.asyncio
    async def test_with_redis_retry_custom_parameters(self):
        """Test retry with custom parameters"""
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Timeout")
        
        with pytest.raises(CacheOperationError):
            await with_redis_retry(
                failing_operation,
                max_retries=1,
                base_delay=0.1,
                max_delay=0.2,
                backoff_factor=1.5
            )
        
        assert call_count == 2  # Initial + 1 retry
    
    @pytest.mark.asyncio
    async def test_with_redis_retry_sync_function(self):
        """Test retry with synchronous function"""
        def sync_operation():
            return "sync_success"
        
        result = await with_redis_retry(sync_operation)
        assert result == "sync_success"


class TestRedisConfigIntegration:
    """Test Redis connection manager with different configurations"""
    
    def test_connection_with_custom_config(self):
        """Test connection manager with custom configuration"""
        config = RedisConfig(
            host="custom-host",
            port=6380,
            max_connections=10,
            socket_timeout=2.0
        )
        
        manager = RedisConnectionManager(config)
        
        assert manager.config.host == "custom-host"
        assert manager.config.port == 6380
        assert manager.config.max_connections == 10
        assert manager.config.socket_timeout == 2.0
    
    def test_connection_kwargs_generation(self):
        """Test connection kwargs generation from config"""
        config = RedisConfig(
            host="test-host",
            port=6379,
            password="test-password",
            socket_timeout=5.0
        )
        
        manager = RedisConnectionManager(config)
        kwargs = config.get_connection_kwargs(db=1)
        
        assert kwargs["host"] == "test-host"
        assert kwargs["port"] == 6379
        assert kwargs["db"] == 1
        assert kwargs["password"] == "test-password"
        assert kwargs["socket_timeout"] == 5.0
    
    def test_pool_kwargs_generation(self):
        """Test connection pool kwargs generation"""
        config = RedisConfig(
            max_connections=25,
            socket_timeout=3.0
        )
        
        pool_kwargs = config.get_pool_kwargs()
        
        assert pool_kwargs["max_connections"] == 25
        assert pool_kwargs["socket_timeout"] == 3.0


@pytest.mark.integration
class TestRedisConnectionIntegration:
    """Integration tests requiring actual Redis instance"""
    
    @pytest.fixture
    def redis_available(self):
        """Check if Redis is available for testing"""
        try:
            client = redis.Redis(host='localhost', port=6379, socket_timeout=1)
            client.ping()
            return True
        except:
            pytest.skip("Redis not available for integration tests")
    
    @pytest.mark.asyncio
    async def test_real_redis_connection(self, redis_available):
        """Test connection to real Redis instance"""
        config = create_test_config()
        manager = RedisConnectionManager(config)
        
        try:
            # Test sync connection
            client = manager.get_connection(0)
            result = client.ping()
            assert result is True
            
            # Test async connection
            async_client = await manager.get_async_connection(0)
            async_result = await async_client.ping()
            assert async_result is True
            
            # Test health check
            health_result = manager.health_check()
            assert health_result is True
            
        finally:
            await manager.aclose()
    
    @pytest.mark.asyncio
    async def test_real_redis_operations(self, redis_available):
        """Test basic Redis operations"""
        config = create_test_config()
        manager = RedisConnectionManager(config)
        
        try:
            client = await manager.get_async_connection(0)
            
            # Test set/get operations
            await client.set("test_key", "test_value", ex=60)
            value = await client.get("test_key")
            assert value == "test_value"
            
            # Test delete operation
            deleted = await client.delete("test_key")
            assert deleted == 1
            
            # Verify key is gone
            value = await client.get("test_key")
            assert value is None
            
        finally:
            await manager.aclose()


if __name__ == "__main__":
    pytest.main([__file__])