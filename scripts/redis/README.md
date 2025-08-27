# Redis Setup for Time-Series Transformer

This directory contains scripts and configurations for setting up Redis caching infrastructure for the Time-Series Transformer project.

## Quick Start

### Windows (PowerShell)
```powershell
# Install and start Redis using Docker
.\scripts\redis\start_redis.ps1 -Action install
.\scripts\redis\start_redis.ps1 -Action start

# Check status
.\scripts\redis\start_redis.ps1 -Action status
```

### Linux/macOS
```bash
# Make script executable and install
chmod +x scripts/redis/install_redis.sh
./scripts/redis/install_redis.sh --docker

# Verify installation
python scripts/redis/verify_redis.py
```

### Docker Compose (All Platforms)
```bash
# Start Redis with Docker Compose
docker-compose -f docker-compose.redis.yml up -d redis

# Check if Redis is running
docker-compose -f docker-compose.redis.yml ps

# View logs
docker-compose -f docker-compose.redis.yml logs redis
```

## Files Overview

- **`docker-compose.redis.yml`** - Docker Compose configuration for Redis
- **`configs/redis/redis.conf`** - Optimized Redis configuration for ML workloads
- **`scripts/redis/install_redis.sh`** - Linux/macOS installation script
- **`scripts/redis/start_redis.ps1`** - Windows PowerShell management script
- **`scripts/redis/verify_redis.py`** - Comprehensive Redis verification script

## Configuration Details

### Redis Databases
- **DB0**: API response cache
- **DB1**: Model predictions cache
- **DB2**: Feature computation cache
- **DB3**: Session and temporary data

### Memory Settings
- **Max Memory**: 2GB
- **Eviction Policy**: allkeys-lru (Least Recently Used)
- **Persistence**: RDB snapshots + AOF logging

### Performance Optimizations
- Connection pooling support
- Optimized for ML workload patterns
- Compression enabled for large objects
- Latency monitoring enabled

## Verification

Run the verification script to ensure Redis is properly configured:

```bash
python scripts/redis/verify_redis.py
```

This will test:
- ✅ Basic connectivity
- ✅ CRUD operations
- ✅ Data structures (Hash, List, Set)
- ✅ Multiple databases
- ✅ ML workload simulation
- ✅ Performance benchmarks

## Connection Details

- **Host**: localhost
- **Port**: 6379
- **Databases**: 0-15 available
- **Authentication**: None (development setup)

## Troubleshooting

### Redis not starting
1. Check if port 6379 is available: `netstat -an | grep 6379`
2. Check Docker status: `docker ps`
3. View Redis logs: `docker logs timeseries-redis`

### Connection issues
1. Verify Redis is running: `redis-cli ping`
2. Check firewall settings
3. Ensure correct host/port configuration

### Performance issues
1. Monitor memory usage: `redis-cli info memory`
2. Check slow queries: `redis-cli slowlog get 10`
3. Review eviction statistics: `redis-cli info stats`

## Production Considerations

For production deployment:
1. Enable authentication (`requirepass`)
2. Configure SSL/TLS encryption
3. Set up Redis Sentinel for high availability
4. Implement proper backup strategies
5. Monitor memory usage and performance metrics

## Integration with Application

The Redis cache will be integrated with:
- FastAPI endpoints for response caching
- ML pipeline for prediction caching
- Data processing for feature caching
- Model inference for result caching

See the main application code for cache manager implementations.