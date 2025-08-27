@echo off
REM test_all.bat

echo RUNNING ALL INTEGRATION TESTS
echo ================================

REM Test 1: Data Pipeline
echo.
echo TEST 1: Data Pipeline
C:/tools/doppler/doppler.exe run -- python tests/integration/test_data_pipeline.py
if %errorlevel% neq 0 (
    echo ERROR: Data Pipeline Failed
    exit /b 1
)

REM Test 2: Model Forward
echo.
echo TEST 2: Model Forward Pass
C:/tools/doppler/doppler.exe run -- python tests/integration/test_model_forward.py
if %errorlevel% neq 0 (
    echo ERROR: Model Forward Failed
    exit /b 1
)

REM Test 3: Training Components
echo.
echo TEST 3: Training Components
C:/tools/doppler/doppler.exe run -- python tests/integration/test_training_components.py
if %errorlevel% neq 0 (
    echo ERROR: Training Components Failed
    exit /b 1
)

REM Test 4: W&B Integration
echo.
echo TEST 4: W&B Integration
C:/tools/doppler/doppler.exe run -- python tests/integration/test_wandb_integration.py
if %errorlevel% neq 0 (
    echo ERROR: W&B Integration Failed
    exit /b 1
)

REM Test 5: End-to-End
echo.
echo TEST 5: End-to-End Pipeline
C:/tools/doppler/doppler.exe run -- python tests/integration/test_end_to_end.py
if %errorlevel% neq 0 (
    echo ERROR: End-to-End Failed
    exit /b 1
)

echo.
echo SUCCESS: ALL INTEGRATION TESTS PASSED!
echo You're ready for Phase 5: Testing ^& Validation!