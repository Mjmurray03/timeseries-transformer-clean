#!/bin/bash

# Redis Installation Script for Time-Series Transformer Project
# Supports multiple platforms and installation methods

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Check if Docker is available
check_docker() {
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Install Redis using Docker (recommended)
install_redis_docker() {
    log_info "Installing Redis using Docker..."
    
    if ! check_docker; then
        log_error "Docker is not available. Please install Docker first."
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p configs/redis
    mkdir -p data/redis
    
    # Copy configuration if it doesn't exist
    if [ ! -f "configs/redis/redis.conf" ]; then
        log_warn "Redis configuration not found. Using default configuration."
        # The redis.conf should already be created by the previous step
    fi
    
    # Start Redis using Docker Compose
    log_info "Starting Redis container..."
    docker-compose -f docker-compose.redis.yml up -d redis
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    for i in {1..30}; do
        if docker exec timeseries-redis redis-cli ping &> /dev/null; then
            log_info "Redis is ready!"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            log_error "Redis failed to start within 30 seconds"
            exit 1
        fi
    done
}

# Install Redis natively on Ubuntu/Debian
install_redis_debian() {
    log_info "Installing Redis on Debian/Ubuntu..."
    
    sudo apt-get update
    sudo apt-get install -y redis-server redis-tools
    
    # Copy custom configuration
    sudo cp configs/redis/redis.conf /etc/redis/redis.conf
    
    # Enable and start Redis service
    sudo systemctl enable redis-server
    sudo systemctl restart redis-server
    
    log_info "Redis installed and started as a system service"
}

# Install Redis natively on CentOS/RHEL/Fedora
install_redis_redhat() {
    log_info "Installing Redis on RedHat/CentOS/Fedora..."
    
    # Install EPEL repository if needed
    if command -v yum &> /dev/null; then
        sudo yum install -y epel-release
        sudo yum install -y redis redis-tools
    else
        sudo dnf install -y redis redis-tools
    fi
    
    # Copy custom configuration
    sudo cp configs/redis/redis.conf /etc/redis.conf
    
    # Enable and start Redis service
    sudo systemctl enable redis
    sudo systemctl restart redis
    
    log_info "Redis installed and started as a system service"
}

# Install Redis on macOS
install_redis_macos() {
    log_info "Installing Redis on macOS..."
    
    if ! command -v brew &> /dev/null; then
        log_error "Homebrew is required but not installed. Please install Homebrew first."
        exit 1
    fi
    
    brew install redis
    
    # Copy custom configuration
    cp configs/redis/redis.conf /usr/local/etc/redis.conf
    
    # Start Redis service
    brew services start redis
    
    log_info "Redis installed and started using Homebrew"
}

# Verify Redis installation
verify_redis() {
    log_info "Verifying Redis installation..."
    
    # Test Redis connection
    if redis-cli ping &> /dev/null; then
        log_info "✓ Redis is responding to ping"
    else
        log_error "✗ Redis is not responding to ping"
        return 1
    fi
    
    # Test basic operations
    redis-cli set test_key "test_value" &> /dev/null
    if [ "$(redis-cli get test_key)" = "test_value" ]; then
        log_info "✓ Redis basic operations working"
        redis-cli del test_key &> /dev/null
    else
        log_error "✗ Redis basic operations failed"
        return 1
    fi
    
    # Check Redis info
    log_info "Redis server information:"
    redis-cli info server | grep -E "(redis_version|os|arch_bits|process_id|tcp_port)"
    
    return 0
}

# Main installation function
main() {
    log_info "Starting Redis installation for Time-Series Transformer project..."
    
    # Parse command line arguments
    INSTALL_METHOD="auto"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker)
                INSTALL_METHOD="docker"
                shift
                ;;
            --native)
                INSTALL_METHOD="native"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--docker|--native]"
                echo "  --docker: Force Docker installation (recommended)"
                echo "  --native: Force native installation"
                echo "  --help:   Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Detect OS
    OS=$(detect_os)
    log_info "Detected OS: $OS"
    
    # Choose installation method
    if [ "$INSTALL_METHOD" = "auto" ]; then
        if check_docker; then
            INSTALL_METHOD="docker"
            log_info "Docker available, using Docker installation"
        else
            INSTALL_METHOD="native"
            log_info "Docker not available, using native installation"
        fi
    fi
    
    # Install Redis based on method and OS
    case $INSTALL_METHOD in
        docker)
            install_redis_docker
            ;;
        native)
            case $OS in
                debian)
                    install_redis_debian
                    ;;
                redhat)
                    install_redis_redhat
                    ;;
                macos)
                    install_redis_macos
                    ;;
                windows)
                    log_error "Native Redis installation on Windows is not supported by this script."
                    log_info "Please use Docker or WSL2 with Linux installation."
                    exit 1
                    ;;
                *)
                    log_error "Unsupported OS for native installation: $OS"
                    log_info "Please use Docker installation instead."
                    exit 1
                    ;;
            esac
            ;;
    esac
    
    # Verify installation
    sleep 2
    if verify_redis; then
        log_info "✅ Redis installation completed successfully!"
        log_info "Redis is running and ready for use."
        
        # Show connection info
        echo ""
        log_info "Connection details:"
        echo "  Host: localhost"
        echo "  Port: 6379"
        echo "  Databases: 0-15 (configured for different cache types)"
        echo ""
        log_info "To connect: redis-cli"
        log_info "To stop: docker-compose -f docker-compose.redis.yml down (Docker) or sudo systemctl stop redis (native)"
    else
        log_error "❌ Redis installation verification failed!"
        exit 1
    fi
}

# Run main function
main "$@"