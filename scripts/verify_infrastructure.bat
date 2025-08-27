@echo off
REM Comprehensive Infrastructure Verification Script - Windows
REM Time-Series Transformer Project

echo 🔍 Time-Series Transformer Infrastructure Verification
echo ==================================================

setlocal enabledelayedexpansion
set failed_checks=0

echo.

REM Check Doppler CLI connection
echo 📡 Checking Doppler CLI connection...
C:\tools\doppler\doppler.exe me >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Doppler connected
) else (
    echo ❌ Doppler not connected
    set /a failed_checks+=1
)

REM Check API keys
echo 🔐 Verifying API keys from Doppler...
set missing_keys=
for %%k in (ALPHA_VANTAGE_API_KEY NEWSAPI_API_KEY HUGGINGFACE_API_KEY WANDB_API_KEY) do (
    C:\tools\doppler\doppler.exe secrets get "%%k" >nul 2>&1
    if !errorlevel! neq 0 (
        set missing_keys=!missing_keys! %%k
    )
)
if "!missing_keys!" == "" (
    echo ✅ All API keys present
) else (
    echo ❌ Missing keys:!missing_keys!
    set /a failed_checks+=1
)

REM Check Python dependencies
echo 📦 Checking Python dependencies...
set missing_deps=
for %%d in (torch pandas numpy wandb redis) do (
    python -c "import %%d" >nul 2>&1
    if !errorlevel! neq 0 (
        set missing_deps=!missing_deps! %%d
    )
)
if "!missing_deps!" == "" (
    echo ✅ All dependencies installed
) else (
    echo ❌ Missing dependencies:!missing_deps!
    echo    Run: pip install -r requirements.txt
    set /a failed_checks+=1
)

REM Check Redis connection
echo 🔴 Checking Redis connection...
python -c "import redis; r=redis.Redis(); r.ping()" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Redis connected
) else (
    echo ⚠️  Redis not running (optional for development)
)

REM Check Docker files
echo 🐳 Checking Docker infrastructure...
if exist "deployment\docker\Dockerfile.training" (
    for %%A in ("deployment\docker\Dockerfile.training") do (
        if %%~zA gtr 100 (
            echo ✅ Docker files present and non-empty
        ) else (
            echo ❌ Docker files empty
            set /a failed_checks+=1
        )
    )
) else (
    echo ❌ Docker files missing
    set /a failed_checks+=1
)

REM Check Git LFS
echo 📦 Checking Git LFS configuration...
if exist ".gitattributes" (
    git lfs track >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ Git LFS configured
    ) else (
        echo ❌ Git LFS not configured
        set /a failed_checks+=1
    )
) else (
    echo ❌ .gitattributes missing
    set /a failed_checks+=1
)

echo.
echo ==================================================
if !failed_checks! equ 0 (
    echo 🎉 All critical infrastructure checks passed!
    exit /b 0
) else (
    echo ❌ !failed_checks! critical check(s) failed
    echo    Please fix the issues above before proceeding
    exit /b 1
)