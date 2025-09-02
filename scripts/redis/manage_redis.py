#!/usr/bin/env python3
"""
Redis service management script for Time-Series Transformer project.

Provides unified Redis management across different platforms with
service control, health monitoring, and configuration deployment.
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RedisServiceManager:
    """Cross-platform Redis service management"""

    def __init__(self, config_path: Optional[str] = None):
        self.platform = platform.system().lower()
        self.config_path = config_path or self._find_config_path()
        self.docker_compose_file = "docker-compose.redis.yml"

        logger.info(f"Initialized Redis service manager for {self.platform}")

    def _find_config_path(self) -> str:
        """Find Redis configuration file"""
        possible_paths = [
            "configs/redis/redis.conf",
            "../configs/redis/redis.conf",
            "../../configs/redis/redis.conf",
            "/etc/redis/redis.conf",
            "/usr/local/etc/redis.conf",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return "configs/redis/redis.conf"  # Default

    def _run_command(
        self, command: List[str], check: bool = True, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run shell command with error handling"""
        try:
            logger.debug(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command, check=check, capture_output=capture_output, text=True, timeout=30
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out: {e}")
            raise

    def _is_docker_available(self) -> bool:
        """Check if Docker is available and running"""
        try:
            result = self._run_command(["docker", "--version"], check=False)
            if result.returncode != 0:
                return False

            result = self._run_command(["docker", "info"], check=False)
            return result.returncode == 0
        except Exception:
            return False

    def _is_redis_running_docker(self) -> bool:
        """Check if Redis is running in Docker"""
        try:
            result = self._run_command(
                ["docker", "ps", "--filter", "name=timeseries-redis", "--format", "{{.Names}}"],
                check=False,
            )

            return "timeseries-redis" in result.stdout
        except Exception:
            return False

    def _is_redis_running_native(self) -> bool:
        """Check if Redis is running natively"""
        try:
            # Try to connect to Redis
            result = self._run_command(["redis-cli", "ping"], check=False)
            return result.returncode == 0 and "PONG" in result.stdout
        except Exception:
            return False

    def is_redis_running(self) -> Tuple[bool, str]:
        """
        Check if Redis is running and return method.

        Returns:
            Tuple of (is_running, method) where method is 'docker', 'native', or 'none'
        """
        if self._is_redis_running_docker():
            return True, "docker"
        elif self._is_redis_running_native():
            return True, "native"
        else:
            return False, "none"

    def start_redis_docker(self) -> bool:
        """Start Redis using Docker Compose"""
        try:
            logger.info("Starting Redis with Docker Compose...")

            # Check if Docker Compose file exists
            if not os.path.exists(self.docker_compose_file):
                logger.error(f"Docker Compose file not found: {self.docker_compose_file}")
                return False

            # Start Redis service
            self._run_command(
                ["docker-compose", "-f", self.docker_compose_file, "up", "-d", "redis"]
            )

            # Wait for Redis to be ready
            logger.info("Waiting for Redis to be ready...")
            for attempt in range(30):
                if self._is_redis_running_docker():
                    # Test Redis connection
                    try:
                        result = self._run_command(
                            ["docker", "exec", "timeseries-redis", "redis-cli", "ping"], check=False
                        )

                        if result.returncode == 0 and "PONG" in result.stdout:
                            logger.info("Redis started successfully with Docker")
                            return True
                    except Exception:
                        pass

                time.sleep(1)

            logger.error("Redis failed to start within 30 seconds")
            return False

        except Exception as e:
            logger.error(f"Failed to start Redis with Docker: {e}")
            return False

    def start_redis_native(self) -> bool:
        """Start Redis natively (platform-specific)"""
        try:
            if self.platform == "linux":
                return self._start_redis_linux()
            elif self.platform == "darwin":  # macOS
                return self._start_redis_macos()
            elif self.platform == "windows":
                return self._start_redis_windows()
            else:
                logger.error(f"Unsupported platform for native Redis: {self.platform}")
                return False
        except Exception as e:
            logger.error(f"Failed to start Redis natively: {e}")
            return False

    def _start_redis_linux(self) -> bool:
        """Start Redis on Linux"""
        try:
            # Try systemctl first
            result = self._run_command(["sudo", "systemctl", "start", "redis-server"], check=False)
            if result.returncode == 0:
                logger.info("Started Redis with systemctl")
                return True

            # Try service command
            result = self._run_command(["sudo", "service", "redis-server", "start"], check=False)
            if result.returncode == 0:
                logger.info("Started Redis with service command")
                return True

            # Try direct redis-server command
            if os.path.exists(self.config_path):
                subprocess.Popen(
                    ["redis-server", self.config_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Wait a moment and check if it started
                time.sleep(2)
                if self._is_redis_running_native():
                    logger.info("Started Redis directly")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to start Redis on Linux: {e}")
            return False

    def _start_redis_macos(self) -> bool:
        """Start Redis on macOS"""
        try:
            # Try brew services
            result = self._run_command(["brew", "services", "start", "redis"], check=False)
            if result.returncode == 0:
                logger.info("Started Redis with brew services")
                return True

            # Try direct redis-server command
            if os.path.exists(self.config_path):
                subprocess.Popen(
                    ["redis-server", self.config_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Wait a moment and check if it started
                time.sleep(2)
                if self._is_redis_running_native():
                    logger.info("Started Redis directly")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to start Redis on macOS: {e}")
            return False

    def _start_redis_windows(self) -> bool:
        """Start Redis on Windows"""
        logger.error("Native Redis startup on Windows is not supported by this script")
        logger.info("Please use Docker method or install Redis manually")
        return False

    def stop_redis_docker(self) -> bool:
        """Stop Redis Docker container"""
        try:
            logger.info("Stopping Redis Docker container...")

            self._run_command(["docker-compose", "-f", self.docker_compose_file, "stop", "redis"])

            logger.info("Redis Docker container stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop Redis Docker container: {e}")
            return False

    def stop_redis_native(self) -> bool:
        """Stop Redis native service"""
        try:
            if self.platform == "linux":
                return self._stop_redis_linux()
            elif self.platform == "darwin":  # macOS
                return self._stop_redis_macos()
            elif self.platform == "windows":
                return self._stop_redis_windows()
            else:
                logger.error(f"Unsupported platform for native Redis: {self.platform}")
                return False
        except Exception as e:
            logger.error(f"Failed to stop Redis natively: {e}")
            return False

    def _stop_redis_linux(self) -> bool:
        """Stop Redis on Linux"""
        try:
            # Try systemctl first
            result = self._run_command(["sudo", "systemctl", "stop", "redis-server"], check=False)
            if result.returncode == 0:
                logger.info("Stopped Redis with systemctl")
                return True

            # Try service command
            result = self._run_command(["sudo", "service", "redis-server", "stop"], check=False)
            if result.returncode == 0:
                logger.info("Stopped Redis with service command")
                return True

            # Try redis-cli shutdown
            result = self._run_command(["redis-cli", "shutdown"], check=False)
            if result.returncode == 0:
                logger.info("Stopped Redis with redis-cli shutdown")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to stop Redis on Linux: {e}")
            return False

    def _stop_redis_macos(self) -> bool:
        """Stop Redis on macOS"""
        try:
            # Try brew services
            result = self._run_command(["brew", "services", "stop", "redis"], check=False)
            if result.returncode == 0:
                logger.info("Stopped Redis with brew services")
                return True

            # Try redis-cli shutdown
            result = self._run_command(["redis-cli", "shutdown"], check=False)
            if result.returncode == 0:
                logger.info("Stopped Redis with redis-cli shutdown")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to stop Redis on macOS: {e}")
            return False

    def _stop_redis_windows(self) -> bool:
        """Stop Redis on Windows"""
        logger.error("Native Redis stop on Windows is not supported by this script")
        return False

    def get_redis_status(self) -> Dict[str, any]:
        """Get comprehensive Redis status"""
        is_running, method = self.is_redis_running()

        status = {
            "running": is_running,
            "method": method,
            "platform": self.platform,
            "config_path": self.config_path,
            "docker_available": self._is_docker_available(),
        }

        if is_running:
            try:
                # Get Redis info
                if method == "docker":
                    result = self._run_command(
                        ["docker", "exec", "timeseries-redis", "redis-cli", "info", "server"],
                        check=False,
                    )
                else:
                    result = self._run_command(["redis-cli", "info", "server"], check=False)

                if result.returncode == 0:
                    # Parse Redis info
                    info_lines = result.stdout.strip().split("\n")
                    redis_info = {}

                    for line in info_lines:
                        if ":" in line and not line.startswith("#"):
                            key, value = line.split(":", 1)
                            redis_info[key] = value

                    status["redis_info"] = redis_info

            except Exception as e:
                logger.warning(f"Failed to get Redis info: {e}")

        return status

    def deploy_configuration(self) -> bool:
        """Deploy Redis configuration"""
        try:
            logger.info("Deploying Redis configuration...")

            # Ensure config directory exists
            config_dir = os.path.dirname(self.config_path)
            os.makedirs(config_dir, exist_ok=True)

            # Check if config file exists
            if not os.path.exists(self.config_path):
                logger.error(f"Redis configuration file not found: {self.config_path}")
                return False

            # For Docker deployment, config is mounted as volume
            if self._is_docker_available():
                logger.info("Configuration will be deployed via Docker volume mount")
                return True

            # For native deployment, copy config to system location
            if self.platform == "linux":
                try:
                    self._run_command(["sudo", "cp", self.config_path, "/etc/redis/redis.conf"])
                    logger.info("Deployed configuration to /etc/redis/redis.conf")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to deploy to system location: {e}")

            elif self.platform == "darwin":  # macOS
                try:
                    self._run_command(["cp", self.config_path, "/usr/local/etc/redis.conf"])
                    logger.info("Deployed configuration to /usr/local/etc/redis.conf")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to deploy to system location: {e}")

            logger.info("Configuration deployment completed")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy Redis configuration: {e}")
            return False


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Redis service management for Time-Series Transformer"
    )

    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status", "deploy-config"],
        help="Action to perform",
    )

    parser.add_argument(
        "--method",
        choices=["auto", "docker", "native"],
        default="auto",
        help="Method to use for Redis management",
    )

    parser.add_argument("--config", help="Path to Redis configuration file")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize service manager
    manager = RedisServiceManager(args.config)

    try:
        if args.action == "status":
            status = manager.get_redis_status()
            print(json.dumps(status, indent=2))

            if status["running"]:
                print(f"\n✅ Redis is running ({status['method']})")
                if "redis_info" in status:
                    redis_info = status["redis_info"]
                    print(f"   Version: {redis_info.get('redis_version', 'unknown')}")
                    print(f"   Port: {redis_info.get('tcp_port', 'unknown')}")
                    print(f"   Uptime: {redis_info.get('uptime_in_seconds', 'unknown')} seconds")
            else:
                print("❌ Redis is not running")

            sys.exit(0 if status["running"] else 1)

        elif args.action == "deploy-config":
            success = manager.deploy_configuration()
            if success:
                print("✅ Configuration deployed successfully")
                sys.exit(0)
            else:
                print("❌ Configuration deployment failed")
                sys.exit(1)

        elif args.action == "start":
            is_running, current_method = manager.is_redis_running()

            if is_running:
                print(f"✅ Redis is already running ({current_method})")
                sys.exit(0)

            # Determine method
            if args.method == "auto":
                if manager._is_docker_available():
                    method = "docker"
                else:
                    method = "native"
            else:
                method = args.method

            print(f"Starting Redis using {method} method...")

            if method == "docker":
                success = manager.start_redis_docker()
            else:
                success = manager.start_redis_native()

            if success:
                print("✅ Redis started successfully")
                sys.exit(0)
            else:
                print("❌ Failed to start Redis")
                sys.exit(1)

        elif args.action == "stop":
            is_running, current_method = manager.is_redis_running()

            if not is_running:
                print("✅ Redis is not running")
                sys.exit(0)

            print(f"Stopping Redis ({current_method})...")

            if current_method == "docker":
                success = manager.stop_redis_docker()
            else:
                success = manager.stop_redis_native()

            if success:
                print("✅ Redis stopped successfully")
                sys.exit(0)
            else:
                print("❌ Failed to stop Redis")
                sys.exit(1)

        elif args.action == "restart":
            print("Restarting Redis...")

            # Stop first
            is_running, current_method = manager.is_redis_running()
            if is_running:
                if current_method == "docker":
                    manager.stop_redis_docker()
                else:
                    manager.stop_redis_native()

                # Wait a moment
                time.sleep(2)

            # Start
            if args.method == "auto":
                if manager._is_docker_available():
                    method = "docker"
                else:
                    method = "native"
            else:
                method = args.method

            if method == "docker":
                success = manager.start_redis_docker()
            else:
                success = manager.start_redis_native()

            if success:
                print("✅ Redis restarted successfully")
                sys.exit(0)
            else:
                print("❌ Failed to restart Redis")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
