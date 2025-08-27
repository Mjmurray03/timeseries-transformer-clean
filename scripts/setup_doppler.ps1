# Doppler Setup Script for Time-Series Transformer Project
Write-Host "Setting up Doppler CLI for Time-Series Transformer Project" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$dopplerPath = "C:\tools\doppler\doppler.exe"

# Check if Doppler is installed
if (-not (Test-Path $dopplerPath)) {
    Write-Host "ERROR: Doppler CLI not found at $dopplerPath" -ForegroundColor Red
    Write-Host "Please ensure Doppler is installed correctly." -ForegroundColor Red
    exit 1
}

# Display version
Write-Host "Doppler CLI version:" -ForegroundColor Green
& $dopplerPath --version
Write-Host ""

# Step 1: Login
Write-Host "Step 1: Login to Doppler" -ForegroundColor Yellow
Write-Host "-------------------------" -ForegroundColor Yellow
Write-Host "This will open your browser for authentication."
Write-Host "Please login with your Doppler account."
Write-Host ""

try {
    & $dopplerPath login -y
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to login to Doppler"
    }
}
catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 2: Setup Project Configuration" -ForegroundColor Yellow
Write-Host "------------------------------------" -ForegroundColor Yellow

# Change to project directory
Set-Location -Path "C:\timeseries-transformer"

Write-Host "Current directory: $(Get-Location)"
Write-Host "Running doppler setup..."
Write-Host ""

& $dopplerPath setup

Write-Host ""
Write-Host "Step 3: Test Configuration" -ForegroundColor Yellow
Write-Host "---------------------------" -ForegroundColor Yellow
Write-Host "Running test script..."
Write-Host ""

& $dopplerPath run -- python scripts\test_doppler.py

Write-Host ""
Write-Host "Doppler setup complete!" -ForegroundColor Green
Read-Host "Press Enter to continue"