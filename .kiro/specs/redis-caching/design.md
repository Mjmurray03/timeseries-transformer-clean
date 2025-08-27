# Design Document

## Overview

This design implements a comprehensive Redis caching layer for the time-series transformer project. The solution provides prediction caching, feature storage, and connection management with proper error handling and monitoring. The design follows the existing project architecture and integrates seamlessly with the FastAPI application and ML pipeline.

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  ML Pipeline    │    │  Data Pipeline  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │Cache Manager│ │    │ │Pred. Cache  │ │    │ │Feature Cache│ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────┐
                    │     Redis Cluster       │
                    │                         │
                    │  ┌─────┐  ┌─────┐      │
                    │  │ DB0 │  │ DB1 │ ...  │
                    │  └─────┘  └─────┘      │
                    └─────────────────────────┘
```

### Redis Configuration Strategy

- **Database Separation**: Use different Redis databases for different data types
  - DB0: API response cache
  - DB1: Model predictions cache  
  - DB2: Feature computation cache
  - DB3: Session and temporary data

- **Memory Management**: Configure Redis with appropriate eviction policies
  - `allkeys-lru` for prediction cache (DB1)
  - `volatile-ttl` for API cache (DB0)
  - `noeviction` for critical feature cache (DB2)

## Components and Interfaces

### 1. Redis Connection Manager

```python
class RedisConnectionManager:
    """Manages Redis connections with pooling and failover"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.pool = None
        self.health_check_interval = 30
    
    async def get_connection(self, db: int = 0) -> Redis:
        """Get Redis connection from pool"""
    
    async def health_check(self) -> bool:
        """Verify Redis connectivity and performance"""
    
    async def close(self):
        """Clean shutdown of connections"""
```

### 2. Cache Manager Interface

```python
class CacheManager(ABC):
    """Abstract base for cache implementations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value"""
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Store value in cache"""
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove key from cache"""
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
```

### 3. Prediction Cache

```python
class PredictionCache(CacheManager):
    """Specialized cache for ML predictions"""
    
    def __init__(self, redis_manager: RedisConnectionManager):
        self.redis = redis_manager
        self.db = 1  # Dedicated DB for predictions
        self.default_ttl = 300  # 5 minutes
    
    async def cache_prediction(
        self, 
        ticker: str, 
        features_hash: str, 
        prediction: Dict[str, Any]
    ) -> bool:
        """Cache model prediction with metadata"""
    
    async def get_prediction(
        self, 
        ticker: str, 
        features_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction if valid"""
    
    def _generate_cache_key(self, ticker: str, features_hash: str) -> str:
        """Generate consistent cache keys"""
```

### 4. Feature Cache

```python
class FeatureCache(CacheManager):
    """Cache for computed technical indicators and features"""
    
    def __init__(self, redis_manager: RedisConnectionManager):
        self.redis = redis_manager
        self.db = 2
        self.default_ttl = 3600  # 1 hour for features
    
    async def cache_features(
        self, 
        ticker: str, 
        date_range: Tuple[str, str],
        features: pd.DataFrame
    ) -> bool:
        """Cache computed features with compression"""
    
    async def get_features(
        self, 
        ticker: str, 
        date_range: Tuple[str, str]
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached features"""
```

## Data Models

### Redis Configuration Model

```python
@dataclass
class RedisConfig:
    """Redis connection and behavior configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Memory settings
    maxmemory: str = "2gb"
    maxmemory_policy: str = "allkeys-lru"
    
    # Persistence settings
    save_intervals: List[Tuple[int, int]] = field(default_factory=lambda: [(900, 1), (300, 10), (60, 10000)])
    
    # Cache TTL settings
    api_cache_ttl: int = 300  # 5 minutes
    prediction_cache_ttl: int = 900  # 15 minutes
    feature_cache_ttl: int = 3600  # 1 hour
```

### Cache Entry Model

```python
@dataclass
class CacheEntry:
    """Standardized cache entry with metadata"""
    data: Any
    created_at: datetime
    ttl: int
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for Redis storage"""
        return {
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "ttl": self.ttl,
            "version": self.version,
            "metadata": self.metadata
        }
```

## Error Handling

### Connection Resilience

```python
class RedisConnectionError(Exception):
    """Redis connection related errors"""
    pass

class CacheOperationError(Exception):
    """Cache operation failures"""
    pass

async def with_redis_retry(operation: Callable, max_retries: int = 3) -> Any:
    """Retry Redis operations with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return await operation()
        except (ConnectionError, TimeoutError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Redis operation failed after {max_retries} attempts: {e}")
                raise CacheOperationError(f"Redis unavailable: {e}")
            
            wait_time = 2 ** attempt
            logger.warning(f"Redis operation failed, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)
```

### Graceful Degradation

```python
class CacheMiddleware:
    """Middleware for graceful cache degradation"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.fallback_enabled = True
    
    async def get_or_compute(
        self, 
        key: str, 
        compute_func: Callable, 
        ttl: int = None
    ) -> Any:
        """Get from cache or compute with fallback"""
        try:
            # Try cache first
            cached = await self.cache.get(key)
            if cached is not None:
                return cached
        except CacheOperationError:
            logger.warning("Cache unavailable, computing directly")
        
        # Compute value
        result = await compute_func()
        
        # Try to cache result
        if self.fallback_enabled:
            try:
                await self.cache.set(key, result, ttl)
            except CacheOperationError:
                logger.warning("Failed to cache computed result")
        
        return result
```

## Testing Strategy

### Unit Tests

1. **Connection Manager Tests**
   - Connection pool creation and management
   - Health check functionality
   - Connection retry logic
   - Graceful shutdown

2. **Cache Manager Tests**
   - Basic CRUD operations
   - TTL handling
   - Key generation consistency
   - Error handling

3. **Serialization Tests**
   - DataFrame serialization/deserialization
   - Complex object caching
   - Compression effectiveness
   - Data integrity verification

### Integration Tests

1. **Redis Integration**
   - End-to-end cache operations
   - Multiple database usage
   - Concurrent access patterns
   - Memory usage under load

2. **Application Integration**
   - FastAPI middleware integration
   - ML pipeline caching
   - Performance improvement verification
   - Fallback behavior testing

### Performance Tests

1. **Latency Benchmarks**
   - Cache hit/miss performance
   - Serialization overhead
   - Network latency impact
   - Concurrent operation throughput

2. **Memory Usage Tests**
   - Cache size limits
   - Eviction policy effectiveness
   - Memory leak detection
   - Compression ratio analysis

## Installation and Deployment

### Docker Deployment

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  redis_data:
```

### Configuration Management

```python
def load_redis_config() -> RedisConfig:
    """Load Redis configuration from environment"""
    return RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD"),
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
        maxmemory=os.getenv("REDIS_MAXMEMORY", "2gb"),
        maxmemory_policy=os.getenv("REDIS_MAXMEMORY_POLICY", "allkeys-lru")
    )
```

### Monitoring Integration

```python
class RedisMetrics:
    """Redis performance metrics collection"""
    
    def __init__(self, redis_manager: RedisConnectionManager):
        self.redis = redis_manager
        
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect Redis performance metrics"""
        info = await self.redis.info()
        return {
            "memory_usage": info.get("used_memory"),
            "connected_clients": info.get("connected_clients"),
            "operations_per_second": info.get("instantaneous_ops_per_sec"),
            "hit_rate": self._calculate_hit_rate(info),
            "evicted_keys": info.get("evicted_keys")
        }
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0
```