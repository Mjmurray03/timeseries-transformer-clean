#!/usr/bin/env python3
"""
Redis Verification Script for Time-Series Transformer Project
Tests Redis installation, configuration, and performance
"""

import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import redis
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RedisVerifier:
    """Comprehensive Redis verification and testing"""

    def __init__(self, host: str = "localhost", port: int = 6379, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self.redis_client = None
        self.async_client = None

    def connect(self) -> bool:
        """Establish Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            # Test connection
            self.redis_client.ping()
            logger.info("✓ Redis connection established")
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"✗ Redis connection failed: {e}")
            return False

    async def connect_async(self) -> bool:
        """Establish async Redis connection"""
        try:
            self.async_client = aioredis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

            # Test connection
            await self.async_client.ping()
            logger.info("✓ Async Redis connection established")
            return True

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"✗ Async Redis connection failed: {e}")
            return False

    def test_basic_operations(self) -> bool:
        """Test basic Redis operations"""
        logger.info("Testing basic Redis operations...")

        try:
            # Test SET/GET
            test_key = "test:basic:key"
            test_value = "test_value_123"

            self.redis_client.set(test_key, test_value)
            retrieved_value = self.redis_client.get(test_key)

            if retrieved_value != test_value:
                logger.error("✗ SET/GET operation failed")
                return False

            # Test DELETE
            self.redis_client.delete(test_key)
            if self.redis_client.exists(test_key):
                logger.error("✗ DELETE operation failed")
                return False

            logger.info("✓ Basic operations (SET/GET/DELETE) working")

            # Test TTL
            ttl_key = "test:ttl:key"
            self.redis_client.setex(ttl_key, 2, "ttl_value")

            if not self.redis_client.exists(ttl_key):
                logger.error("✗ TTL SET operation failed")
                return False

            time.sleep(3)
            if self.redis_client.exists(ttl_key):
                logger.error("✗ TTL expiration failed")
                return False

            logger.info("✓ TTL operations working")
            return True

        except Exception as e:
            logger.error(f"✗ Basic operations failed: {e}")
            return False

    def test_data_structures(self) -> bool:
        """Test Redis data structures"""
        logger.info("Testing Redis data structures...")

        try:
            # Test Hash (for feature storage)
            hash_key = "test:hash:features"
            hash_data = {"rsi": "65.5", "macd": "1.23", "volume": "1000000"}

            self.redis_client.hset(hash_key, mapping=hash_data)
            retrieved_hash = self.redis_client.hgetall(hash_key)

            if retrieved_hash != hash_data:
                logger.error("✗ Hash operations failed")
                return False

            self.redis_client.delete(hash_key)
            logger.info("✓ Hash operations working")

            # Test List (for sequences)
            list_key = "test:list:sequence"
            list_data = ["1.0", "2.0", "3.0", "4.0", "5.0"]

            self.redis_client.lpush(list_key, *list_data)
            retrieved_list = self.redis_client.lrange(list_key, 0, -1)

            if len(retrieved_list) != len(list_data):
                logger.error("✗ List operations failed")
                return False

            self.redis_client.delete(list_key)
            logger.info("✓ List operations working")

            # Test Set (for cache keys)
            set_key = "test:set:cache_keys"
            set_data = {"key1", "key2", "key3"}

            self.redis_client.sadd(set_key, *set_data)
            retrieved_set = self.redis_client.smembers(set_key)

            if retrieved_set != set_data:
                logger.error("✗ Set operations failed")
                return False

            self.redis_client.delete(set_key)
            logger.info("✓ Set operations working")

            return True

        except Exception as e:
            logger.error(f"✗ Data structure tests failed: {e}")
            return False

    def test_multiple_databases(self) -> bool:
        """Test multiple Redis databases"""
        logger.info("Testing multiple Redis databases...")

        try:
            # Test different databases
            databases = [0, 1, 2, 3]  # API, Predictions, Features, Sessions

            for db_num in databases:
                db_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=db_num,
                    decode_responses=True,
                )

                test_key = f"test:db{db_num}:key"
                test_value = f"value_for_db_{db_num}"

                db_client.set(test_key, test_value)
                retrieved_value = db_client.get(test_key)

                if retrieved_value != test_value:
                    logger.error(f"✗ Database {db_num} operations failed")
                    return False

                db_client.delete(test_key)

            logger.info("✓ Multiple database operations working")
            return True

        except Exception as e:
            logger.error(f"✗ Multiple database tests failed: {e}")
            return False

    async def test_performance(self) -> Dict[str, float]:
        """Test Redis performance metrics"""
        logger.info("Testing Redis performance...")

        if not self.async_client:
            await self.connect_async()

        results = {}

        try:
            # Test SET performance
            num_operations = 1000
            start_time = time.time()

            tasks = []
            for i in range(num_operations):
                task = self.async_client.set(f"perf:test:{i}", f"value_{i}")
                tasks.append(task)

            await asyncio.gather(*tasks)
            set_time = time.time() - start_time
            set_ops_per_sec = num_operations / set_time
            results["set_ops_per_sec"] = set_ops_per_sec

            logger.info(f"✓ SET performance: {set_ops_per_sec:.2f} ops/sec")

            # Test GET performance
            start_time = time.time()

            tasks = []
            for i in range(num_operations):
                task = self.async_client.get(f"perf:test:{i}")
                tasks.append(task)

            await asyncio.gather(*tasks)
            get_time = time.time() - start_time
            get_ops_per_sec = num_operations / get_time
            results["get_ops_per_sec"] = get_ops_per_sec

            logger.info(f"✓ GET performance: {get_ops_per_sec:.2f} ops/sec")

            # Test latency
            latencies = []
            for _ in range(100):
                start = time.time()
                await self.async_client.ping()
                latency = (time.time() - start) * 1000  # Convert to ms
                latencies.append(latency)

            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            results["avg_latency_ms"] = avg_latency
            results["max_latency_ms"] = max_latency

            logger.info(f"✓ Average latency: {avg_latency:.2f}ms")
            logger.info(f"✓ Max latency: {max_latency:.2f}ms")

            # Cleanup performance test keys
            keys_to_delete = [f"perf:test:{i}" for i in range(num_operations)]
            if keys_to_delete:
                await self.async_client.delete(*keys_to_delete)

            return results

        except Exception as e:
            logger.error(f"✗ Performance tests failed: {e}")
            return {}

    def get_redis_info(self) -> Dict[str, str]:
        """Get Redis server information"""
        logger.info("Collecting Redis server information...")

        try:
            info = self.redis_client.info()

            # Extract key information
            key_info = {
                "redis_version": info.get("redis_version", "unknown"),
                "os": info.get("os", "unknown"),
                "arch_bits": str(info.get("arch_bits", "unknown")),
                "tcp_port": str(info.get("tcp_port", "unknown")),
                "uptime_in_seconds": str(info.get("uptime_in_seconds", "unknown")),
                "connected_clients": str(info.get("connected_clients", "unknown")),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "maxmemory_human": info.get("maxmemory_human", "unknown"),
                "maxmemory_policy": info.get("maxmemory_policy", "unknown"),
            }

            logger.info("✓ Redis server information collected")
            return key_info

        except Exception as e:
            logger.error(f"✗ Failed to get Redis info: {e}")
            return {}

    def test_ml_workload_simulation(self) -> bool:
        """Simulate ML workload patterns"""
        logger.info("Testing ML workload simulation...")

        try:
            # Simulate prediction caching
            prediction_data = {
                "ticker": "AAPL",
                "prediction": [100.5, 101.2, 99.8, 102.1, 103.0],
                "confidence": [0.85, 0.82, 0.88, 0.79, 0.81],
                "timestamp": str(int(time.time())),
                "model_version": "v1.0.0",
            }

            prediction_key = "prediction:AAPL:hash123"
            self.redis_client.setex(
                prediction_key, 300, json.dumps(prediction_data)  # 5 minutes TTL
            )

            # Retrieve and verify
            cached_prediction = self.redis_client.get(prediction_key)
            if not cached_prediction:
                logger.error("✗ Prediction caching failed")
                return False

            parsed_prediction = json.loads(cached_prediction)
            if parsed_prediction["ticker"] != "AAPL":
                logger.error("✗ Prediction data integrity failed")
                return False

            logger.info("✓ Prediction caching working")

            # Simulate feature caching
            feature_data = {
                "rsi_14": 65.5,
                "macd": 1.23,
                "bb_upper": 105.2,
                "bb_lower": 98.7,
                "volume_sma": 1500000,
            }

            feature_key = "features:AAPL:2024-01-01:2024-01-31"
            self.redis_client.hset(
                feature_key, mapping={k: str(v) for k, v in feature_data.items()}
            )
            self.redis_client.expire(feature_key, 3600)  # 1 hour TTL

            # Retrieve and verify
            cached_features = self.redis_client.hgetall(feature_key)
            if len(cached_features) != len(feature_data):
                logger.error("✗ Feature caching failed")
                return False

            logger.info("✓ Feature caching working")

            # Cleanup
            self.redis_client.delete(prediction_key, feature_key)

            return True

        except Exception as e:
            logger.error(f"✗ ML workload simulation failed: {e}")
            return False

    def cleanup(self):
        """Cleanup connections"""
        if self.redis_client:
            self.redis_client.close()
        if self.async_client:
            asyncio.create_task(self.async_client.close())


async def main():
    """Main verification function"""
    logger.info("Starting Redis verification for Time-Series Transformer project")

    verifier = RedisVerifier()

    # Test results
    results = {
        "connection": False,
        "basic_operations": False,
        "data_structures": False,
        "multiple_databases": False,
        "ml_workload": False,
        "performance": {},
        "server_info": {},
    }

    try:
        # Test connection
        if not verifier.connect():
            logger.error("❌ Redis verification failed - cannot connect")
            return False

        results["connection"] = True

        # Test basic operations
        results["basic_operations"] = verifier.test_basic_operations()

        # Test data structures
        results["data_structures"] = verifier.test_data_structures()

        # Test multiple databases
        results["multiple_databases"] = verifier.test_multiple_databases()

        # Test ML workload simulation
        results["ml_workload"] = verifier.test_ml_workload_simulation()

        # Get server info
        results["server_info"] = verifier.get_redis_info()

        # Test performance
        results["performance"] = await verifier.test_performance()

        # Summary
        passed_tests = sum(
            [
                results["connection"],
                results["basic_operations"],
                results["data_structures"],
                results["multiple_databases"],
                results["ml_workload"],
            ]
        )

        total_tests = 5

        logger.info(f"\n{'='*50}")
        logger.info("REDIS VERIFICATION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")

        if results["server_info"]:
            logger.info(f"Redis version: {results['server_info'].get('redis_version', 'unknown')}")
            logger.info(
                f"Memory used: {results['server_info'].get('used_memory_human', 'unknown')}"
            )
            logger.info(f"Max memory: {results['server_info'].get('maxmemory_human', 'unknown')}")
            logger.info(
                f"Eviction policy: {results['server_info'].get('maxmemory_policy', 'unknown')}"
            )

        if results["performance"]:
            logger.info(
                f"SET performance: {results['performance'].get('set_ops_per_sec', 0):.2f} ops/sec"
            )
            logger.info(
                f"GET performance: {results['performance'].get('get_ops_per_sec', 0):.2f} ops/sec"
            )
            logger.info(f"Average latency: {results['performance'].get('avg_latency_ms', 0):.2f}ms")

        if passed_tests == total_tests:
            logger.info("✅ All Redis verification tests passed!")
            logger.info("Redis is ready for Time-Series Transformer caching workloads")
            return True
        else:
            logger.error("❌ Some Redis verification tests failed")
            return False

    except Exception as e:
        logger.error(f"❌ Redis verification failed with exception: {e}")
        return False

    finally:
        verifier.cleanup()


if __name__ == "__main__":
    # Check if redis package is available
    try:
        import redis
        import redis.asyncio
    except ImportError:
        print("❌ Redis Python package not found. Install with: pip install redis")
        sys.exit(1)

    # Run verification
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
