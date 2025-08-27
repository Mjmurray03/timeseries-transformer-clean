@echo off
echo Setting up Doppler CLI for Time-Series Transformer Project
echo ============================================================
echo.

REM Check if Doppler is installed
C:\tools\doppler\doppler.exe --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Doppler CLI not found at C:\tools\doppler\doppler.exe
    echo Please ensure Doppler is installed correctly.
    exit /b 1
)

echo Doppler CLI version:
C:\tools\doppler\doppler.exe --version
echo.

echo Step 1: Login to Doppler
echo -------------------------
echo This will open your browser for authentication.
echo Please login with your Doppler account.
echo.
C:\tools\doppler\doppler.exe login -y

if %errorlevel% neq 0 (
    echo ERROR: Failed to login to Doppler
    exit /b 1
)

echo.
echo Step 2: Setup Project Configuration
echo ------------------------------------
cd /d C:\timeseries-transformer
C:\tools\doppler\doppler.exe setup

echo.
echo Step 3: Test Configuration
echo ---------------------------
echo Running test script...
C:\tools\doppler\doppler.exe run -- python scripts\test_doppler.py

echo.
echo Doppler setup complete!
pause