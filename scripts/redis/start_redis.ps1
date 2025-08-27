# Redis Management Script for Windows PowerShell
# Time-Series Transformer Project

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "stop", "restart", "status", "install")]
    [string]$Action = "start",
    
    [Parameter(Mandatory=$false)]
    [switch]$UseDocker = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Colors = @{
    Info = "Green"
    Warn = "Yellow" 
    Error = "Red"
    Success = "Cyan"
}

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("Info", "Warn", "Error", "Success")]
        [string]$Level = "Info"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = $Colors[$Level]
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

function Test-DockerAvailable {
    try {
        $null = docker --version 2>$null
        $null = docker info 2>$null
        return $true
    }
    catch {
        return $false
    }
}

function Test-RedisRunning {
    if ($UseDocker) {
        try {
            $container = docker ps --filter "name=timeseries-redis" --format "{{.Names}}" 2>$null
            return $container -eq "timeseries-redis"
        }
        catch {
            return $false
        }
    }
    else {
        # Check if Redis is running on port 6379
        try {
            $connection = Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet
            return $connection
        }
        catch {
            return $false
        }
    }
}

function Install-Redis {
    Write-Log "Installing Redis for Time-Series Transformer project..." -Level Info
    
    if ($UseDocker) {
        if (-not (Test-DockerAvailable)) {
            Write-Log "Docker is not available. Please install Docker Desktop first." -Level Error
            Write-Log "Download from: https://www.docker.com/products/docker-desktop" -Level Info
            exit 1
        }
        
        Write-Log "Using Docker installation method" -Level Info
        
        # Create necessary directories
        $dirs = @("configs\redis", "data\redis", "scripts\redis")
        foreach ($dir in $dirs) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
                Write-Log "Created directory: $dir" -Level Info
            }
        }
        
        # Check if docker-compose file exists
        if (-not (Test-Path "docker-compose.redis.yml")) {
            Write-Log "docker-compose.redis.yml not found. Please ensure it exists." -Level Error
            exit 1
        }
        
        # Pull Redis image
        Write-Log "Pulling Redis Docker image..." -Level Info
        docker pull redis:7-alpine
        
        Write-Log "Redis installation completed!" -Level Success
    }
    else {
        Write-Log "Native Redis installation on Windows requires manual setup." -Level Warn
        Write-Log "Please use one of these options:" -Level Info
        Write-Log "1. Use Docker (recommended): Run with -UseDocker" -Level Info
        Write-Log "2. Use WSL2 with Linux Redis installation" -Level Info
        Write-Log "3. Download Redis for Windows from: https://github.com/microsoftarchive/redis/releases" -Level Info
    }
}

function Start-Redis {
    Write-Log "Starting Redis server..." -Level Info
    
    if ($UseDocker) {
        if (-not (Test-DockerAvailable)) {
            Write-Log "Docker is not available. Cannot start Redis." -Level Error
            exit 1
        }
        
        if (Test-RedisRunning) {
            Write-Log "Redis is already running" -Level Warn
            return
        }
        
        # Start Redis using Docker Compose
        Write-Log "Starting Redis container..." -Level Info
        docker-compose -f docker-compose.redis.yml up -d redis
        
        # Wait for Redis to be ready
        Write-Log "Waiting for Redis to be ready..." -Level Info
        $maxAttempts = 30
        $attempt = 0
        
        do {
            $attempt++
            Start-Sleep -Seconds 1
            
            try {
                $result = docker exec timeseries-redis redis-cli ping 2>$null
                if ($result -eq "PONG") {
                    Write-Log "Redis is ready!" -Level Success
                    break
                }
            }
            catch {
                # Continue waiting
            }
            
            if ($attempt -eq $maxAttempts) {
                Write-Log "Redis failed to start within 30 seconds" -Level Error
                exit 1
            }
        } while ($attempt -lt $maxAttempts)
    }
    else {
        Write-Log "Native Redis startup not implemented for Windows" -Level Error
        Write-Log "Please use Docker method: -UseDocker" -Level Info
        exit 1
    }
}

function Stop-Redis {
    Write-Log "Stopping Redis server..." -Level Info
    
    if ($UseDocker) {
        if (Test-RedisRunning) {
            docker-compose -f docker-compose.redis.yml stop redis
            Write-Log "Redis stopped successfully" -Level Success
        }
        else {
            Write-Log "Redis is not running" -Level Warn
        }
    }
    else {
        Write-Log "Native Redis stop not implemented for Windows" -Level Error
        Write-Log "Please use Docker method: -UseDocker" -Level Info
    }
}

function Restart-Redis {
    Write-Log "Restarting Redis server..." -Level Info
    Stop-Redis
    Start-Sleep -Seconds 2
    Start-Redis
}

function Get-RedisStatus {
    Write-Log "Checking Redis status..." -Level Info
    
    if (Test-RedisRunning) {
        Write-Log "✓ Redis is running" -Level Success
        
        if ($UseDocker) {
            # Get container info
            $containerInfo = docker ps --filter "name=timeseries-redis" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            Write-Host $containerInfo
            
            # Test Redis connection
            try {
                $info = docker exec timeseries-redis redis-cli info server 2>$null
                if ($info) {
                    Write-Log "✓ Redis is responding to commands" -Level Success
                    
                    # Extract key information
                    $lines = $info -split "`n"
                    foreach ($line in $lines) {
                        if ($line -match "redis_version:(.+)" -or 
                            $line -match "os:(.+)" -or 
                            $line -match "tcp_port:(.+)" -or
                            $line -match "uptime_in_seconds:(.+)") {
                            Write-Host "  $line"
                        }
                    }
                }
            }
            catch {
                Write-Log "✗ Redis is not responding to commands" -Level Error
            }
        }
    }
    else {
        Write-Log "✗ Redis is not running" -Level Error
    }
    
    # Show connection details
    Write-Log "Connection details:" -Level Info
    Write-Host "  Host: localhost"
    Write-Host "  Port: 6379"
    Write-Host "  Databases: 0-15 (configured for different cache types)"
}

function Test-RedisConnection {
    Write-Log "Testing Redis connection..." -Level Info
    
    if (-not (Test-RedisRunning)) {
        Write-Log "Redis is not running" -Level Error
        return $false
    }
    
    try {
        if ($UseDocker) {
            # Test ping
            $ping = docker exec timeseries-redis redis-cli ping 2>$null
            if ($ping -ne "PONG") {
                Write-Log "✗ Redis ping failed" -Level Error
                return $false
            }
            Write-Log "✓ Redis ping successful" -Level Success
            
            # Test basic operations
            docker exec timeseries-redis redis-cli set test_key "test_value" | Out-Null
            $value = docker exec timeseries-redis redis-cli get test_key 2>$null
            
            if ($value -eq "test_value") {
                Write-Log "✓ Redis basic operations working" -Level Success
                docker exec timeseries-redis redis-cli del test_key | Out-Null
                return $true
            }
            else {
                Write-Log "✗ Redis basic operations failed" -Level Error
                return $false
            }
        }
    }
    catch {
        Write-Log "✗ Redis connection test failed: $($_.Exception.Message)" -Level Error
        return $false
    }
}

# Main execution
try {
    Write-Log "Redis Management Script for Time-Series Transformer" -Level Info
    Write-Log "Action: $Action, UseDocker: $UseDocker" -Level Info
    
    switch ($Action.ToLower()) {
        "install" {
            Install-Redis
        }
        "start" {
            Start-Redis
            Test-RedisConnection
        }
        "stop" {
            Stop-Redis
        }
        "restart" {
            Restart-Redis
            Test-RedisConnection
        }
        "status" {
            Get-RedisStatus
            Test-RedisConnection
        }
        default {
            Write-Log "Unknown action: $Action" -Level Error
            Write-Log "Available actions: install, start, stop, restart, status" -Level Info
            exit 1
        }
    }
    
    Write-Log "Operation completed successfully!" -Level Success
}
catch {
    Write-Log "Operation failed: $($_.Exception.Message)" -Level Error
    exit 1
}