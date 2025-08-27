#!/bin/bash

# Redis Service Management Script for Time-Series Transformer
# Provides systemd service integration for Linux systems

set -e

# Configuration
SERVICE_NAME="timeseries-redis"
REDIS_USER="redis"
REDIS_GROUP="redis"
REDIS_CONFIG="/etc/redis/timeseries-redis.conf"
REDIS_DATA_DIR="/var/lib/redis/timeseries"
REDIS_LOG_DIR="/var/log/redis"
REDIS_PID_FILE="/var/run/redis/timeseries-redis.pid"

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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Create Redis user and group
create_redis_user() {
    if ! id "$REDIS_USER" &>/dev/null; then
        log_info "Creating Redis user: $REDIS_USER"
        useradd --system --home-dir /var/lib/redis --shell /bin/false $REDIS_USER
    else
        log_info "Redis user already exists: $REDIS_USER"
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating Redis directories..."
    
    # Data directory
    mkdir -p "$REDIS_DATA_DIR"
    chown $REDIS_USER:$REDIS_GROUP "$REDIS_DATA_DIR"
    chmod 750 "$REDIS_DATA_DIR"
    
    # Log directory
    mkdir -p "$REDIS_LOG_DIR"
    chown $REDIS_USER:$REDIS_GROUP "$REDIS_LOG_DIR"
    chmod 750 "$REDIS_LOG_DIR"
    
    # PID directory
    mkdir -p "$(dirname "$REDIS_PID_FILE")"
    chown $REDIS_USER:$REDIS_GROUP "$(dirname "$REDIS_PID_FILE")"
    chmod 755 "$(dirname "$REDIS_PID_FILE")"
    
    # Config directory
    mkdir -p "$(dirname "$REDIS_CONFIG")"
    chmod 755 "$(dirname "$REDIS_CONFIG")"
}

# Install Redis configuration
install_config() {
    log_info "Installing Redis configuration..."
    
    # Find source config file
    SOURCE_CONFIG=""
    POSSIBLE_CONFIGS=(
        "configs/redis/redis.conf"
        "../configs/redis/redis.conf"
        "../../configs/redis/redis.conf"
        "./redis.conf"
    )
    
    for config in "${POSSIBLE_CONFIGS[@]}"; do
        if [[ -f "$config" ]]; then
            SOURCE_CONFIG="$config"
            break
        fi
    done
    
    if [[ -z "$SOURCE_CONFIG" ]]; then
        log_error "Redis configuration file not found"
        log_info "Please ensure configs/redis/redis.conf exists"
        exit 1
    fi
    
    # Copy and modify config for service
    cp "$SOURCE_CONFIG" "$REDIS_CONFIG"
    
    # Update config for service deployment
    sed -i "s|^dir .*|dir $REDIS_DATA_DIR|g" "$REDIS_CONFIG"
    sed -i "s|^pidfile .*|pidfile $REDIS_PID_FILE|g" "$REDIS_CONFIG"
    sed -i "s|^logfile .*|logfile $REDIS_LOG_DIR/redis.log|g" "$REDIS_CONFIG"
    sed -i "s|^daemonize no|daemonize yes|g" "$REDIS_CONFIG"
    
    # Set permissions
    chown root:$REDIS_GROUP "$REDIS_CONFIG"
    chmod 640 "$REDIS_CONFIG"
    
    log_info "Redis configuration installed to $REDIS_CONFIG"
}

# Create systemd service file
create_systemd_service() {
    log_info "Creating systemd service file..."
    
    cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Time-Series Transformer Redis Server
Documentation=https://redis.io/documentation
After=network.target
Wants=network-online.target

[Service]
Type=notify
ExecStart=/usr/bin/redis-server $REDIS_CONFIG
ExecStop=/bin/kill -s QUIT \$MAINPID
ExecReload=/bin/kill -s HUP \$MAINPID
TimeoutStopSec=0
Restart=always
User=$REDIS_USER
Group=$REDIS_GROUP
RuntimeDirectory=redis
RuntimeDirectoryMode=0755

# Security settings
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=$REDIS_DATA_DIR $REDIS_LOG_DIR
CapabilityBoundingSet=CAP_SETGID CAP_SETUID CAP_SYS_RESOURCE
MemoryDenyWriteExecute=true
ProtectKernelModules=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictNamespaces=true
LockPersonality=true

# Resource limits
LimitNOFILE=65535
LimitNPROC=65535

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    
    log_info "Systemd service created: ${SERVICE_NAME}.service"
}

# Install Redis service
install_service() {
    log_info "Installing Redis service for Time-Series Transformer..."
    
    check_root
    
    # Check if Redis is installed
    if ! command -v redis-server &> /dev/null; then
        log_error "Redis server is not installed"
        log_info "Please install Redis first:"
        log_info "  Ubuntu/Debian: sudo apt-get install redis-server"
        log_info "  CentOS/RHEL: sudo yum install redis"
        exit 1
    fi
    
    create_redis_user
    create_directories
    install_config
    create_systemd_service
    
    log_info "Redis service installation completed"
    log_info "Use the following commands to manage the service:"
    log_info "  Start:   sudo systemctl start ${SERVICE_NAME}"
    log_info "  Stop:    sudo systemctl stop ${SERVICE_NAME}"
    log_info "  Enable:  sudo systemctl enable ${SERVICE_NAME}"
    log_info "  Status:  sudo systemctl status ${SERVICE_NAME}"
}

# Uninstall Redis service
uninstall_service() {
    log_info "Uninstalling Redis service..."
    
    check_root
    
    # Stop and disable service
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        systemctl stop "${SERVICE_NAME}"
        log_info "Stopped ${SERVICE_NAME} service"
    fi
    
    if systemctl is-enabled --quiet "${SERVICE_NAME}"; then
        systemctl disable "${SERVICE_NAME}"
        log_info "Disabled ${SERVICE_NAME} service"
    fi
    
    # Remove service file
    if [[ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]]; then
        rm "/etc/systemd/system/${SERVICE_NAME}.service"
        systemctl daemon-reload
        log_info "Removed systemd service file"
    fi
    
    # Remove configuration (with confirmation)
    if [[ -f "$REDIS_CONFIG" ]]; then
        read -p "Remove Redis configuration file? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm "$REDIS_CONFIG"
            log_info "Removed Redis configuration"
        fi
    fi
    
    # Remove data directory (with confirmation)
    if [[ -d "$REDIS_DATA_DIR" ]]; then
        read -p "Remove Redis data directory? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$REDIS_DATA_DIR"
            log_info "Removed Redis data directory"
        fi
    fi
    
    log_info "Redis service uninstallation completed"
}

# Start Redis service
start_service() {
    check_root
    
    log_info "Starting Redis service..."
    systemctl start "${SERVICE_NAME}"
    
    # Wait for service to start
    sleep 2
    
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        log_info "✅ Redis service started successfully"
        
        # Test connection
        if redis-cli ping &>/dev/null; then
            log_info "✅ Redis is responding to connections"
        else
            log_warn "⚠️  Redis service is running but not responding"
        fi
    else
        log_error "❌ Failed to start Redis service"
        systemctl status "${SERVICE_NAME}" --no-pager
        exit 1
    fi
}

# Stop Redis service
stop_service() {
    check_root
    
    log_info "Stopping Redis service..."
    systemctl stop "${SERVICE_NAME}"
    
    if ! systemctl is-active --quiet "${SERVICE_NAME}"; then
        log_info "✅ Redis service stopped successfully"
    else
        log_error "❌ Failed to stop Redis service"
        exit 1
    fi
}

# Restart Redis service
restart_service() {
    check_root
    
    log_info "Restarting Redis service..."
    systemctl restart "${SERVICE_NAME}"
    
    # Wait for service to start
    sleep 2
    
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        log_info "✅ Redis service restarted successfully"
    else
        log_error "❌ Failed to restart Redis service"
        systemctl status "${SERVICE_NAME}" --no-pager
        exit 1
    fi
}

# Enable Redis service
enable_service() {
    check_root
    
    log_info "Enabling Redis service for automatic startup..."
    systemctl enable "${SERVICE_NAME}"
    
    if systemctl is-enabled --quiet "${SERVICE_NAME}"; then
        log_info "✅ Redis service enabled for automatic startup"
    else
        log_error "❌ Failed to enable Redis service"
        exit 1
    fi
}

# Disable Redis service
disable_service() {
    check_root
    
    log_info "Disabling Redis service automatic startup..."
    systemctl disable "${SERVICE_NAME}"
    
    if ! systemctl is-enabled --quiet "${SERVICE_NAME}"; then
        log_info "✅ Redis service disabled from automatic startup"
    else
        log_error "❌ Failed to disable Redis service"
        exit 1
    fi
}

# Show service status
show_status() {
    log_info "Redis service status:"
    
    if systemctl is-active --quiet "${SERVICE_NAME}"; then
        echo "✅ Service Status: Active (Running)"
    else
        echo "❌ Service Status: Inactive (Stopped)"
    fi
    
    if systemctl is-enabled --quiet "${SERVICE_NAME}"; then
        echo "✅ Auto-start: Enabled"
    else
        echo "❌ Auto-start: Disabled"
    fi
    
    echo ""
    systemctl status "${SERVICE_NAME}" --no-pager || true
    
    # Test Redis connection
    echo ""
    log_info "Testing Redis connection..."
    if redis-cli ping &>/dev/null; then
        echo "✅ Redis Connection: OK"
        
        # Show Redis info
        echo ""
        log_info "Redis Server Information:"
        redis-cli info server | grep -E "(redis_version|os|arch_bits|process_id|tcp_port|uptime_in_seconds)"
    else
        echo "❌ Redis Connection: Failed"
    fi
}

# Show usage information
show_usage() {
    echo "Redis Service Management Script for Time-Series Transformer"
    echo ""
    echo "Usage: $0 {install|uninstall|start|stop|restart|enable|disable|status}"
    echo ""
    echo "Commands:"
    echo "  install    - Install Redis as a systemd service"
    echo "  uninstall  - Remove Redis systemd service"
    echo "  start      - Start Redis service"
    echo "  stop       - Stop Redis service"
    echo "  restart    - Restart Redis service"
    echo "  enable     - Enable Redis service for automatic startup"
    echo "  disable    - Disable Redis service automatic startup"
    echo "  status     - Show Redis service status"
    echo ""
    echo "Examples:"
    echo "  sudo $0 install    # Install Redis service"
    echo "  sudo $0 start      # Start Redis"
    echo "  sudo $0 enable     # Enable auto-start"
    echo "  $0 status          # Check status (no sudo needed)"
}

# Main script logic
case "${1:-}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    enable)
        enable_service
        ;;
    disable)
        disable_service
        ;;
    status)
        show_status
        ;;
    *)
        show_usage
        exit 1
        ;;
esac