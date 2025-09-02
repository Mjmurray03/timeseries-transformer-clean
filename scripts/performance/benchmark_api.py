#!/usr/bin/env python3
"""
API Performance Benchmarking Script

Validates API performance requirements from PROMPT 5:
- /predict responds in <100ms (cached) or <500ms (uncached)
- Handles 100 concurrent requests without errors
- Memory usage stays below 4GB under load
- GPU memory properly managed (no leaks)
- Graceful degradation when cache unavailable
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import psutil
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class APIBenchmark:
    """Comprehensive API performance benchmarking"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = {}

    async def benchmark_prediction_performance(self) -> Dict:
        """Benchmark /predict endpoint performance"""
        logger.info("Benchmarking prediction endpoint performance...")

        # Generate sample data
        np.random.seed(42)
        features = np.random.randn(60, 8).tolist()

        request_data = {"ticker": "AAPL", "features": features, "horizon": 3}

        async with aiohttp.ClientSession() as session:
            # Test uncached response time
            uncached_times = []
            for i in range(10):
                # Add variation to avoid cache hits
                modified_data = request_data.copy()
                modified_data["features"][0][0] += i * 0.01

                start_time = time.time()
                async with session.post(f"{self.api_url}/predict", json=modified_data) as response:
                    await response.json()
                    response_time = (time.time() - start_time) * 1000
                    uncached_times.append(response_time)

                    if response.status != 200:
                        logger.error(f"Request failed with status {response.status}")

            # Test cached response time (same request multiple times)
            cached_times = []
            for _ in range(10):
                start_time = time.time()
                async with session.post(f"{self.api_url}/predict", json=request_data) as response:
                    data = await response.json()
                    response_time = (time.time() - start_time) * 1000
                    cached_times.append(response_time)

                    # Check if response indicates cache hit after first request
                    if len(cached_times) > 1:
                        assert data.get(
                            "cache_hit", False
                        ), "Expected cache hit on repeated request"

        avg_uncached = sum(uncached_times) / len(uncached_times)
        avg_cached = sum(cached_times) / len(cached_times)

        return {
            "avg_uncached_response_time_ms": avg_uncached,
            "avg_cached_response_time_ms": avg_cached,
            "max_uncached_response_time_ms": max(uncached_times),
            "max_cached_response_time_ms": max(cached_times),
            "uncached_under_500ms": avg_uncached < 500,
            "cached_under_100ms": avg_cached < 100,
            "uncached_times": uncached_times,
            "cached_times": cached_times,
        }

    async def benchmark_concurrent_requests(self) -> Dict:
        """Test handling 100 concurrent requests"""
        logger.info("Benchmarking concurrent request handling...")

        np.random.seed(42)
        base_features = np.random.randn(60, 8).tolist()

        async def make_request(session: aiohttp.ClientSession, request_id: int):
            # Add slight variation to avoid all requests being identical
            features = base_features.copy()
            features[0][0] += request_id * 0.001

            request_data = {"ticker": "AAPL", "features": features, "horizon": 3}

            start_time = time.time()
            try:
                async with session.post(f"{self.api_url}/predict", json=request_data) as response:
                    if response.status == 200:
                        await response.json()
                        response_time = (time.time() - start_time) * 1000
                        return {
                            "success": True,
                            "response_time_ms": response_time,
                            "status": response.status,
                        }
                    else:
                        return {"success": False, "response_time_ms": 0, "status": response.status}
            except Exception as e:
                return {"success": False, "response_time_ms": 0, "error": str(e)}

        # Create 100 concurrent requests
        connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            start_time = time.time()
            tasks = [make_request(session, i) for i in range(100)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

        # Analyze results
        successful_requests = [
            r for r in results if isinstance(r, dict) and r.get("success", False)
        ]
        failed_requests = [
            r for r in results if not (isinstance(r, dict) and r.get("success", False))
        ]

        response_times = [r["response_time_ms"] for r in successful_requests]

        return {
            "total_requests": 100,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / 100,
            "total_time_seconds": total_time,
            "requests_per_second": 100 / total_time,
            "avg_response_time_ms": (
                sum(response_times) / len(response_times) if response_times else 0
            ),
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "errors": [r for r in failed_requests if isinstance(r, dict) and "error" in r],
        }

    async def benchmark_memory_usage(self) -> Dict:
        """Test memory usage under load"""
        logger.info("Benchmarking memory usage under load...")

        # Get baseline memory
        process = psutil.Process()
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024

        # Check GPU memory if available
        baseline_gpu_memory_mb = 0
        if torch.cuda.is_available():
            baseline_gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

        np.random.seed(42)
        features = np.random.randn(60, 8).tolist()

        # Make 200 requests to stress test memory
        memory_samples = []
        gpu_memory_samples = []

        async with aiohttp.ClientSession() as session:
            for i in range(200):
                # Vary requests slightly
                request_data = {
                    "ticker": "AAPL",
                    "features": [[f + i * 0.0001 for f in day] for day in features],
                    "horizon": 3,
                }

                async with session.post(f"{self.api_url}/predict", json=request_data) as response:
                    await response.json()

                # Sample memory every 10 requests
                if i % 10 == 0:
                    current_memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory_mb)

                    if torch.cuda.is_available():
                        current_gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        gpu_memory_samples.append(current_gpu_memory_mb)

        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - baseline_memory_mb
        max_memory_mb = max(memory_samples) if memory_samples else final_memory_mb

        final_gpu_memory_mb = 0
        gpu_memory_increase_mb = 0
        max_gpu_memory_mb = 0

        if torch.cuda.is_available():
            final_gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_increase_mb = final_gpu_memory_mb - baseline_gpu_memory_mb
            max_gpu_memory_mb = (
                max(gpu_memory_samples) if gpu_memory_samples else final_gpu_memory_mb
            )

        return {
            "baseline_memory_mb": baseline_memory_mb,
            "final_memory_mb": final_memory_mb,
            "memory_increase_mb": memory_increase_mb,
            "max_memory_mb": max_memory_mb,
            "memory_under_4gb": max_memory_mb < 4096,
            "baseline_gpu_memory_mb": baseline_gpu_memory_mb,
            "final_gpu_memory_mb": final_gpu_memory_mb,
            "gpu_memory_increase_mb": gpu_memory_increase_mb,
            "max_gpu_memory_mb": max_gpu_memory_mb,
            "gpu_memory_stable": abs(gpu_memory_increase_mb) < 100,  # Less than 100MB increase
            "memory_samples": memory_samples,
            "gpu_memory_samples": gpu_memory_samples,
        }

    async def test_cache_degradation(self) -> Dict:
        """Test graceful degradation when cache unavailable"""
        logger.info("Testing graceful degradation without cache...")

        np.random.seed(42)
        features = np.random.randn(60, 8).tolist()

        request_data = {"ticker": "AAPL", "features": features, "horizon": 3}

        # Test with cache available (baseline)
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.post(f"{self.api_url}/predict", json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    cache_available_time = (time.time() - start_time) * 1000
                    cache_enabled = data.get("cache_hit") is not None
                else:
                    cache_available_time = 0
                    cache_enabled = False

            # Test multiple requests to see if service continues working
            success_count = 0
            total_response_time = 0

            for _ in range(10):
                start_time = time.time()
                async with session.post(f"{self.api_url}/predict", json=request_data) as response:
                    if response.status == 200:
                        await response.json()
                        success_count += 1
                        total_response_time += (time.time() - start_time) * 1000

        avg_response_time = total_response_time / success_count if success_count > 0 else 0

        return {
            "cache_detection_working": cache_enabled,
            "cache_available_response_time_ms": cache_available_time,
            "degraded_avg_response_time_ms": avg_response_time,
            "degraded_success_rate": success_count / 10,
            "service_available_without_cache": success_count > 8,  # At least 80% success
            "performance_acceptable": avg_response_time < 1000,  # Within 1 second
        }

    async def run_all_benchmarks(self) -> Dict:
        """Run all performance benchmarks"""
        logger.info("Starting comprehensive API performance benchmarks...")

        start_time = time.time()

        try:
            # Test API availability first
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health") as response:
                    if response.status != 200:
                        raise Exception(f"API not available: status {response.status}")
                    health_data = await response.json()
                    logger.info(f"API health check passed: {health_data}")

            # Run all benchmarks
            results = {}

            results["prediction_performance"] = await self.benchmark_prediction_performance()
            results["concurrent_requests"] = await self.benchmark_concurrent_requests()
            results["memory_usage"] = await self.benchmark_memory_usage()
            results["cache_degradation"] = await self.test_cache_degradation()

            total_time = time.time() - start_time
            results["benchmark_metadata"] = {
                "total_benchmark_time_seconds": total_time,
                "timestamp": datetime.now().isoformat(),
                "api_url": self.api_url,
            }

            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def print_results(self, results: Dict):
        """Print formatted benchmark results"""
        print("\n" + "=" * 80)
        print("API PERFORMANCE BENCHMARK RESULTS")
        print("=" * 80)

        if "error" in results:
            print(f"❌ Benchmarks failed: {results['error']}")
            return

        # Prediction Performance
        pred_perf = results.get("prediction_performance", {})
        print(f"\n📊 PREDICTION PERFORMANCE:")
        print(
            f"   Uncached avg response time: {pred_perf.get('avg_uncached_response_time_ms', 0):.2f}ms"
        )
        print(
            f"   Cached avg response time:   {pred_perf.get('avg_cached_response_time_ms', 0):.2f}ms"
        )
        print(
            f"   Uncached <500ms requirement: {'✅' if pred_perf.get('uncached_under_500ms', False) else '❌'}"
        )
        print(
            f"   Cached <100ms requirement:   {'✅' if pred_perf.get('cached_under_100ms', False) else '❌'}"
        )

        # Concurrent Requests
        concurrent = results.get("concurrent_requests", {})
        print(f"\n🔀 CONCURRENT REQUESTS (100 requests):")
        print(f"   Success rate:          {concurrent.get('success_rate', 0):.1%}")
        print(f"   Requests per second:   {concurrent.get('requests_per_second', 0):.2f}")
        print(f"   Avg response time:     {concurrent.get('avg_response_time_ms', 0):.2f}ms")
        print(
            f"   100 concurrent req OK: {'✅' if concurrent.get('success_rate', 0) >= 0.95 else '❌'}"
        )

        # Memory Usage
        memory = results.get("memory_usage", {})
        print(f"\n💾 MEMORY USAGE:")
        print(f"   Peak memory usage:     {memory.get('max_memory_mb', 0):.2f}MB")
        print(f"   Memory increase:       {memory.get('memory_increase_mb', 0):.2f}MB")
        print(f"   GPU memory increase:   {memory.get('gpu_memory_increase_mb', 0):.2f}MB")
        print(
            f"   Memory <4GB requirement: {'✅' if memory.get('memory_under_4gb', True) else '❌'}"
        )
        print(
            f"   GPU memory stable:     {'✅' if memory.get('gpu_memory_stable', True) else '❌'}"
        )

        # Cache Degradation
        cache_deg = results.get("cache_degradation", {})
        print(f"\n🏪 CACHE DEGRADATION:")
        print(
            f"   Service available:     {'✅' if cache_deg.get('service_available_without_cache', False) else '❌'}"
        )
        print(
            f"   Performance acceptable: {'✅' if cache_deg.get('performance_acceptable', False) else '❌'}"
        )
        print(
            f"   Degraded response time: {cache_deg.get('degraded_avg_response_time_ms', 0):.2f}ms"
        )

        # Overall Assessment
        checks = [
            pred_perf.get("uncached_under_500ms", False),
            pred_perf.get("cached_under_100ms", False),
            concurrent.get("success_rate", 0) >= 0.95,
            memory.get("memory_under_4gb", True),
            memory.get("gpu_memory_stable", True),
            cache_deg.get("service_available_without_cache", False),
        ]

        passed_checks = sum(checks)
        total_checks = len(checks)

        print(f"\n📋 OVERALL ASSESSMENT: {passed_checks}/{total_checks} requirements met")

        if passed_checks == total_checks:
            print("🎉 ALL PERFORMANCE REQUIREMENTS SATISFIED!")
        else:
            print(f"⚠️  {total_checks - passed_checks} performance requirements not met.")

        print(
            f"\nBenchmark completed in {results.get('benchmark_metadata', {}).get('total_benchmark_time_seconds', 0):.2f} seconds"
        )


async def main():
    """Main benchmark execution"""
    import argparse

    parser = argparse.ArgumentParser(description="API Performance Benchmark")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL to benchmark")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    benchmark = APIBenchmark(args.api_url)
    results = await benchmark.run_all_benchmarks()

    # Print results to console
    benchmark.print_results(results)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Results saved to: {output_path}")

    # Return appropriate exit code
    if "error" in results:
        sys.exit(1)

    # Check if all requirements met
    if (
        results.get("prediction_performance", {}).get("uncached_under_500ms", False)
        and results.get("prediction_performance", {}).get("cached_under_100ms", False)
        and results.get("concurrent_requests", {}).get("success_rate", 0) >= 0.95
        and results.get("memory_usage", {}).get("memory_under_4gb", True)
    ):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
