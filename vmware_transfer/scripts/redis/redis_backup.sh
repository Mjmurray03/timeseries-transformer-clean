#!/bin/bash

# Redis Backup and Restore Script for Time-Series Transformer
# Provides automated backup, restore, and maintenance operations

set -e

# Configuration
BACKUP_DIR="/var/backups/redis/timeseries"
REDIS_DATA_DIR="/var/lib/redis/timeseries"
REDIS_CONFIG="/etc/redis/timeseries-redis.conf"
RETENTION_DAYS=30
COMPRESS_BACKUPS=true
BACKUP_PREFIX="timeseries-redis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check if Redis is running
is_redis_running() {
    if redis-cli ping &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Create backup directory
create_backup_dir() {
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_info "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        
        # Set appropriate permissions
        if [[ $EUID -eq 0 ]]; then
            chown redis:redis "$BACKUP_DIR"
            chmod 750 "$BACKUP_DIR"
        fi
    fi
}

# Generate backup filename
generate_backup_filename() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local filename="${BACKUP_PREFIX}_${timestamp}"
    
    if [[ "$COMPRESS_BACKUPS" == "true" ]]; then
        filename="${filename}.tar.gz"
    else
        filename="${filename}.tar"
    fi
    
    echo "$filename"
}

# Create Redis backup
create_backup() {
    log_info "Starting Redis backup..."
    
    create_backup_dir
    
    local backup_filename=$(generate_backup_filename)
    local backup_path="${BACKUP_DIR}/${backup_filename}"
    local temp_dir=$(mktemp -d)
    
    # Create backup metadata
    local metadata_file="${temp_dir}/backup_metadata.json"
    cat > "$metadata_file" << EOF
{
    "backup_date": "$(date -Iseconds)",
    "redis_version": "$(redis-cli info server | grep redis_version | cut -d: -f2 | tr -d '\r')",
    "backup_type": "full",
    "source_host": "$(hostname)",
    "data_dir": "$REDIS_DATA_DIR",
    "config_file": "$REDIS_CONFIG"
}
EOF
    
    # Trigger Redis BGSAVE if Redis is running
    if is_redis_running; then
        log_info "Triggering Redis background save..."
        redis-cli bgsave
        
        # Wait for BGSAVE to complete
        log_info "Waiting for background save to complete..."
        while [[ "$(redis-cli lastsave)" == "$(redis-cli lastsave)" ]]; do
            sleep 1
        done
        
        # Wait a bit more to ensure file is written
        sleep 2
        
        log_info "Background save completed"
    else
        log_warn "Redis is not running, backing up existing data files"
    fi
    
    # Copy Redis data files
    if [[ -d "$REDIS_DATA_DIR" ]]; then
        log_info "Copying Redis data files..."
        cp -r "$REDIS_DATA_DIR" "${temp_dir}/data"
    else
        log_warn "Redis data directory not found: $REDIS_DATA_DIR"
        mkdir -p "${temp_dir}/data"
    fi
    
    # Copy Redis configuration
    if [[ -f "$REDIS_CONFIG" ]]; then
        log_info "Copying Redis configuration..."
        cp "$REDIS_CONFIG" "${temp_dir}/redis.conf"
    else
        log_warn "Redis configuration file not found: $REDIS_CONFIG"
    fi
    
    # Create backup archive
    log_info "Creating backup archive: $backup_filename"
    cd "$temp_dir"
    
    if [[ "$COMPRESS_BACKUPS" == "true" ]]; then
        tar -czf "$backup_path" .
    else
        tar -cf "$backup_path" .
    fi
    
    # Cleanup temp directory
    rm -rf "$temp_dir"
    
    # Verify backup
    if [[ -f "$backup_path" ]]; then
        local backup_size=$(du -h "$backup_path" | cut -f1)
        log_info "✅ Backup created successfully: $backup_filename ($backup_size)"
        
        # Test backup integrity
        if [[ "$COMPRESS_BACKUPS" == "true" ]]; then
            if tar -tzf "$backup_path" >/dev/null 2>&1; then
                log_info "✅ Backup integrity verified"
            else
                log_error "❌ Backup integrity check failed"
                return 1
            fi
        else
            if tar -tf "$backup_path" >/dev/null 2>&1; then
                log_info "✅ Backup integrity verified"
            else
                log_error "❌ Backup integrity check failed"
                return 1
            fi
        fi
        
        echo "$backup_path"
        return 0
    else
        log_error "❌ Backup creation failed"
        return 1
    fi
}

# List available backups
list_backups() {
    log_info "Available Redis backups:"
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_warn "Backup directory does not exist: $BACKUP_DIR"
        return 1
    fi
    
    local backups=($(ls -1 "$BACKUP_DIR"/${BACKUP_PREFIX}_*.tar* 2>/dev/null | sort -r))
    
    if [[ ${#backups[@]} -eq 0 ]]; then
        log_warn "No backups found in $BACKUP_DIR"
        return 1
    fi
    
    echo ""
    printf "%-30s %-15s %-20s\n" "BACKUP FILE" "SIZE" "DATE"
    printf "%-30s %-15s %-20s\n" "----------" "----" "----"
    
    for backup in "${backups[@]}"; do
        local filename=$(basename "$backup")
        local size=$(du -h "$backup" | cut -f1)
        local date=$(stat -c %y "$backup" | cut -d' ' -f1,2 | cut -d'.' -f1)
        
        printf "%-30s %-15s %-20s\n" "$filename" "$size" "$date"
    done
    
    echo ""
}

# Restore Redis from backup
restore_backup() {
    local backup_file="$1"
    
    if [[ -z "$backup_file" ]]; then
        log_error "Backup file not specified"
        return 1
    fi
    
    # Check if backup file exists
    if [[ ! -f "$backup_file" ]]; then
        # Try to find it in backup directory
        local full_path="${BACKUP_DIR}/${backup_file}"
        if [[ -f "$full_path" ]]; then
            backup_file="$full_path"
        else
            log_error "Backup file not found: $backup_file"
            return 1
        fi
    fi
    
    log_info "Restoring Redis from backup: $(basename "$backup_file")"
    
    # Check if Redis is running
    if is_redis_running; then
        log_warn "Redis is currently running"
        read -p "Stop Redis to perform restore? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Stopping Redis..."
            if command -v systemctl &>/dev/null; then
                sudo systemctl stop timeseries-redis || redis-cli shutdown
            else
                redis-cli shutdown
            fi
            sleep 2
        else
            log_error "Cannot restore while Redis is running"
            return 1
        fi
    fi
    
    # Create temporary directory for extraction
    local temp_dir=$(mktemp -d)
    
    # Extract backup
    log_info "Extracting backup..."
    cd "$temp_dir"
    
    if [[ "$backup_file" == *.tar.gz ]]; then
        tar -xzf "$backup_file"
    else
        tar -xf "$backup_file"
    fi
    
    # Verify backup contents
    if [[ ! -f "backup_metadata.json" ]]; then
        log_error "Invalid backup file: missing metadata"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Show backup information
    log_info "Backup information:"
    cat "backup_metadata.json" | python3 -m json.tool
    
    # Confirm restore
    echo ""
    read -p "Proceed with restore? This will overwrite current data (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        rm -rf "$temp_dir"
        return 0
    fi
    
    # Backup current data (if exists)
    if [[ -d "$REDIS_DATA_DIR" ]]; then
        local current_backup="${BACKUP_DIR}/pre_restore_$(date +%Y%m%d_%H%M%S).tar.gz"
        log_info "Backing up current data to: $(basename "$current_backup")"
        tar -czf "$current_backup" -C "$(dirname "$REDIS_DATA_DIR")" "$(basename "$REDIS_DATA_DIR")"
    fi
    
    # Restore data directory
    if [[ -d "data" ]]; then
        log_info "Restoring Redis data..."
        
        # Remove current data directory
        if [[ -d "$REDIS_DATA_DIR" ]]; then
            rm -rf "$REDIS_DATA_DIR"
        fi
        
        # Create parent directory
        mkdir -p "$(dirname "$REDIS_DATA_DIR")"
        
        # Copy restored data
        cp -r "data" "$REDIS_DATA_DIR"
        
        # Set permissions
        if [[ $EUID -eq 0 ]]; then
            chown -R redis:redis "$REDIS_DATA_DIR"
            chmod -R 750 "$REDIS_DATA_DIR"
        fi
    fi
    
    # Restore configuration (optional)
    if [[ -f "redis.conf" ]]; then
        read -p "Restore Redis configuration? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Restoring Redis configuration..."
            
            # Backup current config
            if [[ -f "$REDIS_CONFIG" ]]; then
                cp "$REDIS_CONFIG" "${REDIS_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
            fi
            
            # Copy restored config
            cp "redis.conf" "$REDIS_CONFIG"
            
            # Set permissions
            if [[ $EUID -eq 0 ]]; then
                chown root:redis "$REDIS_CONFIG"
                chmod 640 "$REDIS_CONFIG"
            fi
        fi
    fi
    
    # Cleanup temp directory
    rm -rf "$temp_dir"
    
    log_info "✅ Restore completed successfully"
    log_info "You can now start Redis to use the restored data"
    
    return 0
}

# Clean old backups
cleanup_backups() {
    log_info "Cleaning up old backups (older than $RETENTION_DAYS days)..."
    
    if [[ ! -d "$BACKUP_DIR" ]]; then
        log_warn "Backup directory does not exist: $BACKUP_DIR"
        return 0
    fi
    
    local deleted_count=0
    
    # Find and delete old backups
    while IFS= read -r -d '' backup_file; do
        local filename=$(basename "$backup_file")
        log_info "Deleting old backup: $filename"
        rm "$backup_file"
        ((deleted_count++))
    done < <(find "$BACKUP_DIR" -name "${BACKUP_PREFIX}_*.tar*" -type f -mtime +$RETENTION_DAYS -print0)
    
    if [[ $deleted_count -gt 0 ]]; then
        log_info "✅ Deleted $deleted_count old backup(s)"
    else
        log_info "No old backups to delete"
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    if [[ -z "$backup_file" ]]; then
        log_error "Backup file not specified"
        return 1
    fi
    
    # Check if backup file exists
    if [[ ! -f "$backup_file" ]]; then
        # Try to find it in backup directory
        local full_path="${BACKUP_DIR}/${backup_file}"
        if [[ -f "$full_path" ]]; then
            backup_file="$full_path"
        else
            log_error "Backup file not found: $backup_file"
            return 1
        fi
    fi
    
    log_info "Verifying backup: $(basename "$backup_file")"
    
    # Test archive integrity
    if [[ "$backup_file" == *.tar.gz ]]; then
        if tar -tzf "$backup_file" >/dev/null 2>&1; then
            log_info "✅ Archive integrity: OK"
        else
            log_error "❌ Archive integrity: FAILED"
            return 1
        fi
    else
        if tar -tf "$backup_file" >/dev/null 2>&1; then
            log_info "✅ Archive integrity: OK"
        else
            log_error "❌ Archive integrity: FAILED"
            return 1
        fi
    fi
    
    # Extract and verify contents
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    if [[ "$backup_file" == *.tar.gz ]]; then
        tar -xzf "$backup_file"
    else
        tar -xf "$backup_file"
    fi
    
    # Check for required files
    local checks_passed=0
    local total_checks=0
    
    # Check metadata
    ((total_checks++))
    if [[ -f "backup_metadata.json" ]]; then
        log_info "✅ Metadata file: Present"
        ((checks_passed++))
        
        # Validate JSON
        if python3 -m json.tool "backup_metadata.json" >/dev/null 2>&1; then
            log_info "✅ Metadata format: Valid JSON"
        else
            log_warn "⚠️  Metadata format: Invalid JSON"
        fi
    else
        log_error "❌ Metadata file: Missing"
    fi
    
    # Check data directory
    ((total_checks++))
    if [[ -d "data" ]]; then
        log_info "✅ Data directory: Present"
        ((checks_passed++))
        
        # Check for Redis files
        if [[ -f "data/dump.rdb" ]] || [[ -f "data/appendonly.aof" ]]; then
            log_info "✅ Redis data files: Found"
        else
            log_warn "⚠️  Redis data files: Not found (empty backup?)"
        fi
    else
        log_error "❌ Data directory: Missing"
    fi
    
    # Check configuration
    ((total_checks++))
    if [[ -f "redis.conf" ]]; then
        log_info "✅ Configuration file: Present"
        ((checks_passed++))
    else
        log_warn "⚠️  Configuration file: Missing"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
    
    # Summary
    log_info "Verification summary: $checks_passed/$total_checks checks passed"
    
    if [[ $checks_passed -eq $total_checks ]]; then
        log_info "✅ Backup verification: PASSED"
        return 0
    else
        log_warn "⚠️  Backup verification: PASSED with warnings"
        return 0
    fi
}

# Show usage information
show_usage() {
    echo "Redis Backup and Restore Script for Time-Series Transformer"
    echo ""
    echo "Usage: $0 {backup|restore|list|cleanup|verify} [options]"
    echo ""
    echo "Commands:"
    echo "  backup                    - Create a new backup"
    echo "  restore <backup_file>     - Restore from backup"
    echo "  list                      - List available backups"
    echo "  cleanup                   - Remove old backups"
    echo "  verify <backup_file>      - Verify backup integrity"
    echo ""
    echo "Examples:"
    echo "  $0 backup                                    # Create backup"
    echo "  $0 list                                      # List backups"
    echo "  $0 restore timeseries-redis_20240315_120000.tar.gz  # Restore backup"
    echo "  $0 verify latest_backup.tar.gz               # Verify backup"
    echo "  $0 cleanup                                   # Clean old backups"
    echo ""
    echo "Configuration:"
    echo "  BACKUP_DIR=$BACKUP_DIR"
    echo "  RETENTION_DAYS=$RETENTION_DAYS"
    echo "  COMPRESS_BACKUPS=$COMPRESS_BACKUPS"
}

# Main script logic
case "${1:-}" in
    backup)
        create_backup
        ;;
    restore)
        restore_backup "$2"
        ;;
    list)
        list_backups
        ;;
    cleanup)
        cleanup_backups
        ;;
    verify)
        verify_backup "$2"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac