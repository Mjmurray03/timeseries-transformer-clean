import redis
import hashlib
import pickle
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class PredictionCache:
    """Redis-based prediction cache with TTL management"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}/{redis_db}")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None
    
    def generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate deterministic cache key from request"""
        # Create stable hash from request content
        content_parts = [
            request.ticker,
            str(request.horizon),
            # Convert features to string for hashing
            str(sorted([tuple(row) for row in request.features]))
        ]
        content = "|".join(content_parts)
        
        # Generate SHA-256 hash
        cache_key = hashlib.sha256(content.encode('utf-8')).hexdigest()
        return f"prediction:{cache_key}"
    
    def get(self, key: str) -> Optional[PredictionResponse]:
        """Get cached prediction"""
        if not self.redis_client:
            return None
            
        try:
            start_time = time.time()
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                # Deserialize cached response
                cached_dict = pickle.loads(cached_data)
                
                # Update metadata to indicate cache hit
                cached_dict['metadata']['cache_hit'] = True
                cached_dict['metadata']['cache_retrieval_time_ms'] = (time.time() - start_time) * 1000
                
                response = PredictionResponse(**cached_dict)
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return response
            
            logger.debug(f"Cache miss for key: {key[:16]}...")
            return None
            
        except (redis.RedisError, pickle.UnpicklingError, Exception) as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, response: PredictionResponse, ttl: int = 300) -> bool:
        """Cache prediction response with TTL"""
        if not self.redis_client:
            return False
            
        try:
            # Convert response to dict for serialization
            response_dict = response.dict()
            
            # Serialize response
            serialized_data = pickle.dumps(response_dict)
            
            # Set with TTL
            success = self.redis_client.setex(key, ttl, serialized_data)
            
            if success:
                logger.debug(f"Cached response for key: {key[:16]}... with TTL: {ttl}s")
            
            return bool(success)
            
        except (redis.RedisError, pickle.PicklingError, Exception) as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached entry"""
        if not self.redis_client:
            return False
            
        try:
            deleted = self.redis_client.delete(key)
            logger.debug(f"Deleted cache key: {key[:16]}...")
            return bool(deleted)
        except redis.RedisError as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        if not self.redis_client:
            return 0
            
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching pattern: {pattern}")
                return deleted
            return 0
        except redis.RedisError as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"enabled": False, "status": "Redis connection unavailable"}
            
        try:
            info = self.redis_client.info()
            stats = {
                "enabled": True,
                "status": "connected",
                "used_memory": info.get('used_memory_human', 'unknown'),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "connected_clients": info.get('connected_clients', 0),
                "total_commands_processed": info.get('total_commands_processed', 0),
            }
            
            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total_requests = hits + misses
            
            if total_requests > 0:
                stats["hit_rate"] = hits / total_requests
            else:
                stats["hit_rate"] = 0.0
                
            return stats
            
        except redis.RedisError as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": False, "status": f"Error: {e}"}
    
    def health_check(self) -> bool:
        """Check if cache is healthy"""
        if not self.redis_client:
            return False
            
        try:
            # Simple ping test
            response = self.redis_client.ping()
            return response is True
        except redis.RedisError:
            return False
    
    def flush_all(self) -> bool:
        """Flush all cached data (use with caution)"""
        if not self.redis_client:
            return False
            
        try:
            self.redis_client.flushdb()
            logger.warning("Flushed all cache data")
            return True
        except redis.RedisError as e:
            logger.error(f"Cache flush error: {e}")
            return False


class ModelCache:
    """In-memory cache for model instances and preprocessing artifacts"""
    
    def __init__(self, max_size: int = 10):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.max_size = max_size
        logger.info(f"Initialized model cache with max size: {max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached model or artifact"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            logger.debug(f"Model cache hit for: {key}")
            return self.cache[key]
        
        logger.debug(f"Model cache miss for: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Cache model or artifact with LRU eviction"""
        # Evict least recently used items if cache is full
        while len(self.cache) >= self.max_size:
            # Find least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.cache.pop(lru_key, None)
            self.access_times.pop(lru_key, None)
            logger.debug(f"Evicted LRU item: {lru_key}")
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
        logger.debug(f"Cached item: {key}")
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        if key in self.cache:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
            logger.debug(f"Removed cached item: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached items"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cleared model cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "usage_percent": (len(self.cache) / self.max_size) * 100 if self.max_size > 0 else 0,
            "items": list(self.cache.keys())
        }


# Global cache instances (initialized in main.py)
prediction_cache: Optional[PredictionCache] = None
model_cache: Optional[ModelCache] = None


def get_prediction_cache() -> Optional[PredictionCache]:
    """Get global prediction cache instance"""
    return prediction_cache


def get_model_cache() -> Optional[ModelCache]:
    """Get global model cache instance"""
    return model_cache


def initialize_caches(redis_host: str = 'localhost', redis_port: int = 6379, 
                     redis_db: int = 0, model_cache_size: int = 10):
    """Initialize global cache instances"""
    global prediction_cache, model_cache
    
    prediction_cache = PredictionCache(redis_host, redis_port, redis_db)
    model_cache = ModelCache(model_cache_size)
    
    logger.info("Cache system initialized")