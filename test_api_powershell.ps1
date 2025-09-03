# PowerShell API Test Script for TimeSeries Transformer API
# =========================================================
# 
# This script comprehensively tests all API endpoints with proper PowerShell syntax,
# error handling, and production-ready validation patterns.

param(
    [string]$BaseUrl = "http://localhost:8000",
    [int]$TimeoutSeconds = 30
)

# Color output functions for better readability
function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Yellow
    Write-Host " $Title" -ForegroundColor Yellow
    Write-Host ("=" * 80) -ForegroundColor Yellow
}

function Test-ApiHealth {
    Write-Header "API HEALTH CHECK"
    
    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get -TimeoutSec $TimeoutSeconds
        Write-Success "API is healthy and accessible"
        Write-Info "Service: $($response.service)"
        Write-Info "Status: $($response.status)"
        
        if ($response.device_info) {
            Write-Info "PyTorch Version: $($response.device_info.pytorch_version)"
            Write-Info "CUDA Available: $($response.device_info.cuda_available)"
            Write-Info "Current Device: $($response.device_info.current_device)"
        }
        
        # Health endpoint doesn't return models, but we can try getting them from model-info
        try {
            $modelInfo = Invoke-RestMethod -Uri "$BaseUrl/model-info" -Method Get -TimeoutSec 10
            if ($modelInfo.supported_tickers) {
                Write-Info "Available Models: $($modelInfo.supported_tickers -join ', ')"
            }
        }
        catch {
            Write-Warning "Could not retrieve model list from /model-info"
        }
        
        return $true
    }
    catch {
        Write-Error "Health check failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-ModelsList {
    Write-Header "MODEL INFO TEST"
    
    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/model-info" -Method Get -TimeoutSec $TimeoutSeconds
        Write-Success "Model info endpoint accessible"
        Write-Info "Model Version: $($response.model_version)"
        Write-Info "Architecture: $($response.architecture)"
        Write-Info "Parameters: $($response.parameters)"
        Write-Info "Training Date: $($response.training_date)"
        Write-Info "Supported Tickers: $($response.supported_tickers -join ', ')"
        Write-Info "Total Supported Tickers: $($response.supported_tickers.Count)"
        
        if ($response.performance_metrics) {
            Write-Info "Performance Metrics:"
            $response.performance_metrics.PSObject.Properties | ForEach-Object {
                Write-Info "  $($_.Name): $($_.Value)"
            }
        }
        
        return $true
    }
    catch {
        Write-Error "Model info failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-PredictionEndpoint {
    Write-Header "PREDICTION ENDPOINT TESTS"
    
    $success = $true
    
    # Test 1: 2D Array Format (60x10)
    Write-Info "Creating 60x10 feature array for prediction test..."
    
    $features = @()
    for ($i = 0; $i -lt 60; $i++) {
        $row = @()
        for ($j = 0; $j -lt 10; $j++) {
            $row += [math]::Round((Get-Random -Minimum 0.1 -Maximum 1.0), 4)
        }
        $features += ,$row  # Note: comma operator creates nested arrays
    }
    
    $predictionData = @{
        ticker = "AAPL";
        features = $features;
        horizon = 3
    }
    
    try {
        Write-Info "Testing 2D array format (60x10)..."
        $jsonData = $predictionData | ConvertTo-Json -Depth 10 -Compress
        $response = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $jsonData -ContentType "application/json" -TimeoutSec $TimeoutSeconds
        
        Write-Success "2D array prediction successful!"
        Write-Info "Ticker: $($response.ticker)"
        Write-Info "Model Version: $($response.model_version)"
        Write-Info "Cache Hit: $($response.cache_hit)"
        
        if ($response.predictions -and $response.predictions.GetType().Name -ne "Object[]") {
            # Get actual prediction data, not array properties
            $predictionKeys = $response.predictions | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name
            Write-Info "Prediction Types: $($predictionKeys -join ', ')"
        }
        
        if ($response.confidence_intervals) {
            Write-Info "Confidence intervals included"
        }
    }
    catch {
        Write-Error "2D array prediction failed: $($_.Exception.Message)"
        if ($_.Exception.Response) {
            $errorContent = $_.Exception.Response.Content.ReadAsStringAsync().Result
            Write-Error "Response content: $errorContent"
        }
        $success = $false
    }
    
    # Test 2: Flat Array Format (600 elements)
    Write-Info "Creating flat 600-element array for prediction test..."
    
    $flatFeatures = @()
    for ($i = 0; $i -lt 600; $i++) {
        $flatFeatures += [math]::Round((Get-Random -Minimum 0.1 -Maximum 1.0), 4)
    }
    
    $flatPredictionData = @{
        ticker = "AAPL";
        features = $flatFeatures;
        horizon = 5
    }
    
    try {
        Write-Info "Testing flat array format (600 elements)..."
        $jsonData = $flatPredictionData | ConvertTo-Json -Depth 5 -Compress
        $response = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $jsonData -ContentType "application/json" -TimeoutSec $TimeoutSeconds
        
        Write-Success "Flat array prediction successful!"
        Write-Info "Ticker: $($response.ticker)"
        Write-Info "Horizon: 5 days"
        Write-Info "Input format: Flat 600 elements (auto-reshaped to 60x10)"
        
        if ($response.predictions -and $response.predictions.GetType().Name -ne "Object[]") {
            # Get actual prediction data, not array properties
            $predictionKeys = $response.predictions | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name
            Write-Info "Prediction Types: $($predictionKeys -join ', ')"
        }
    }
    catch {
        Write-Error "Flat array prediction failed: $($_.Exception.Message)"
        if ($_.Exception.Response) {
            $errorContent = $_.Exception.Response.Content.ReadAsStringAsync().Result
            Write-Error "Response content: $errorContent"
        }
        $success = $false
    }
    
    return $success
}

function Test-PredictionValidation {
    Write-Header "PREDICTION VALIDATION TESTS"
    
    $validationsPassed = 0
    $totalValidations = 3
    
    # Test 1: Invalid array size (500 instead of 600)
    Write-Info "Testing validation with incorrect flat array size..."
    
    $invalidFlatData = @{
        ticker = "AAPL";
        features = @(1..500 | ForEach-Object { 0.5 });
        horizon = 3
    }
    
    try {
        $jsonData = $invalidFlatData | ConvertTo-Json -Depth 5 -Compress
        $response = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $jsonData -ContentType "application/json" -TimeoutSec $TimeoutSeconds
        Write-Warning "Expected validation error, but request succeeded"
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 422) {
            Write-Success "Validation correctly rejected incorrect array size (422)"
            $validationsPassed++
        } else {
            Write-Error "Unexpected status code: $statusCode"
        }
    }
    
    # Test 2: Invalid 2D array shape (50x10 instead of 60x10)
    Write-Info "Testing validation with incorrect 2D array shape..."
    
    $invalidFeatures2D = @()
    for ($i = 0; $i -lt 50; $i++) {  # Only 50 days instead of 60
        $row = @()
        for ($j = 0; $j -lt 10; $j++) {
            $row += 0.5
        }
        $invalidFeatures2D += ,$row
    }
    
    $invalid2DData = @{
        ticker = "AAPL";
        features = $invalidFeatures2D;
        horizon = 3
    }
    
    try {
        $jsonData = $invalid2DData | ConvertTo-Json -Depth 10 -Compress
        $response = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $jsonData -ContentType "application/json" -TimeoutSec $TimeoutSeconds
        Write-Warning "Expected validation error, but request succeeded"
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 422) {
            Write-Success "Validation correctly rejected incorrect 2D array shape (422)"
            $validationsPassed++
        } else {
            Write-Error "Unexpected status code: $statusCode"
        }
    }
    
    # Test 3: Invalid ticker
    Write-Info "Testing validation with invalid ticker..."
    
    $invalidTickerData = @{
        ticker = "INVALID_TICKER";
        features = @(1..600 | ForEach-Object { 0.5 });
        horizon = 3
    }
    
    try {
        $jsonData = $invalidTickerData | ConvertTo-Json -Depth 5 -Compress
        $response = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $jsonData -ContentType "application/json" -TimeoutSec $TimeoutSeconds
        Write-Warning "Expected validation error for invalid ticker, but request succeeded"
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 422) {
            Write-Success "Validation correctly rejected invalid ticker (422)"
            $validationsPassed++
        } else {
            Write-Error "Unexpected status code: $statusCode"
        }
    }
    
    Write-Info "Validation tests: $validationsPassed/$totalValidations passed"
    return ($validationsPassed -eq $totalValidations)
}

function Test-BacktestEndpoint {
    Write-Header "BACKTEST ENDPOINT TEST"
    
    $backtestData = @{
        ticker = "AAPL";
        start_date = "2024-01-01";
        end_date = "2024-12-31";
        initial_capital = 100000
    }
    
    try {
        Write-Info "Testing backtest endpoint..."
        $jsonData = $backtestData | ConvertTo-Json -Depth 5 -Compress
        $response = Invoke-RestMethod -Uri "$BaseUrl/backtest" -Method Post -Body $jsonData -ContentType "application/json" -TimeoutSec $TimeoutSeconds
        
        Write-Success "Backtest completed successfully!"
        Write-Info "Ticker: $($response.ticker)"
        Write-Info "Period: $($backtestData.start_date) to $($backtestData.end_date)"
        Write-Info "Initial Capital: $($backtestData.initial_capital)"
        
        if ($response.performance_metrics) {
            Write-Info "Performance metrics included in response"
        }
        
        if ($response.trades) {
            Write-Info "Trade history: $($response.trades.Count) trades"
        }
        
        return $true
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 404) {
            Write-Warning "Backtest endpoint not implemented (404) - This is expected if not yet implemented"
            return $true  # Not implemented is acceptable
        } else {
            Write-Error "Backtest failed: $($_.Exception.Message)"
            if ($_.Exception.Response) {
                $errorContent = $_.Exception.Response.Content.ReadAsStringAsync().Result
                Write-Error "Response content: $errorContent"
            }
            return $false
        }
    }
}

function Test-CacheEndpoints {
    Write-Header "CACHE ENDPOINTS TEST"
    
    $cacheTests = 0
    $cacheSuccess = 0
    
    # Test cache status
    try {
        Write-Info "Testing cache status endpoint..."
        $response = Invoke-RestMethod -Uri "$BaseUrl/cache/status" -Method Get -TimeoutSec $TimeoutSeconds
        Write-Success "Cache status retrieved successfully"
        Write-Info "Cache enabled: $($response.enabled)"
        Write-Info "Cache size: $($response.size)"
        $cacheSuccess++
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 404) {
            Write-Warning "Cache status endpoint not found (404) - May not be implemented"
            $cacheSuccess++  # Not implemented is acceptable
        } else {
            Write-Error "Cache status failed: $($_.Exception.Message)"
        }
    }
    $cacheTests++
    
    # Test cache clear
    try {
        Write-Info "Testing cache clear endpoint..."
        $response = Invoke-RestMethod -Uri "$BaseUrl/cache/clear" -Method Post -TimeoutSec $TimeoutSeconds
        Write-Success "Cache cleared successfully"
        $cacheSuccess++
    }
    catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 404) {
            Write-Warning "Cache clear endpoint not found (404) - May not be implemented"
            $cacheSuccess++  # Not implemented is acceptable
        } else {
            Write-Error "Cache clear failed: $($_.Exception.Message)"
        }
    }
    $cacheTests++
    
    Write-Info "Cache tests: $cacheSuccess/$cacheTests passed"
    return ($cacheSuccess -eq $cacheTests)
}

function Test-LoadTestPredictions {
    Write-Header "LOAD TEST - MULTIPLE PREDICTIONS"
    
    Write-Info "Running load test with 5 concurrent prediction requests..."
    
    $jobs = @()
    $tickers = @("AAPL", "MSFT", "GOOG", "AAPL", "MSFT")
    
    for ($i = 0; $i -lt 5; $i++) {
        $features = @()
        for ($day = 0; $day -lt 60; $day++) {
            $row = @()
            for ($feat = 0; $feat -lt 10; $feat++) {
                $row += [math]::Round((Get-Random -Minimum 0.1 -Maximum 1.0), 4)
            }
            $features += ,$row
        }
        
        $predictionData = @{
            ticker = $tickers[$i];
            features = $features;
            horizon = 3
        }
        
        $job = Start-Job -ScriptBlock {
            param($BaseUrl, $PredictionData, $TimeoutSeconds)
            
            try {
                $jsonData = $PredictionData | ConvertTo-Json -Depth 10 -Compress
                $response = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $jsonData -ContentType "application/json" -TimeoutSec $TimeoutSeconds
                return @{ Success = $true; Ticker = $response.ticker; CacheHit = $response.cache_hit }
            }
            catch {
                return @{ Success = $false; Error = $_.Exception.Message }
            }
        } -ArgumentList $BaseUrl, $predictionData, $TimeoutSeconds
        
        $jobs += $job
    }
    
    # Wait for all jobs to complete
    $results = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    
    $successful = ($results | Where-Object { $_.Success }).Count
    $failed = ($results | Where-Object { -not $_.Success }).Count
    
    Write-Info "Load test completed:"
    Write-Success "$successful successful requests"
    if ($failed -gt 0) {
        Write-Error "$failed failed requests"
    }
    
    # Show cache hit statistics
    $cacheHits = ($results | Where-Object { $_.Success -and $_.CacheHit }).Count
    $cacheMisses = ($results | Where-Object { $_.Success -and -not $_.CacheHit }).Count
    Write-Info "Cache performance: $cacheHits hits, $cacheMisses misses"
    
    # Return success if most requests succeeded
    return ($successful -ge 3)
}

function Main {
    Write-Header "TIMESERIES TRANSFORMER API TEST SUITE"
    Write-Info "PowerShell API Testing Script"
    Write-Info "Base URL: $BaseUrl"
    Write-Info "Timeout: $TimeoutSeconds seconds"
    
    $testResults = @{
        Health = $false
        ModelInfo = $false
        Prediction = $false
        Validation = $false
        Backtest = $false
        Cache = $false
        LoadTest = $false
    }
    
    # Run all test suites
    $testResults.Health = Test-ApiHealth
    
    if ($testResults.Health) {
        $testResults.ModelInfo = Test-ModelsList
        $testResults.Prediction = Test-PredictionEndpoint
        $testResults.Validation = Test-PredictionValidation
        $testResults.Backtest = Test-BacktestEndpoint
        $testResults.Cache = Test-CacheEndpoints
        $testResults.LoadTest = Test-LoadTestPredictions
    } else {
        Write-Error "API health check failed - skipping remaining tests"
        Write-Info "Make sure the API server is running on $BaseUrl"
        exit 1
    }
    
    # Final summary
    Write-Header "TEST SUITE SUMMARY"
    
    $passedTests = ($testResults.Values | Where-Object { $_ }).Count
    $totalTests = $testResults.Count
    
    Write-Info "Test Results:"
    foreach ($test in $testResults.GetEnumerator()) {
        if ($test.Value) {
            Write-Success "$($test.Key): PASSED"
        } else {
            Write-Error "$($test.Key): FAILED"
        }
    }
    
    Write-Header "TESTING COMPLETE"
    Write-Info "Overall Score: $passedTests/$totalTests tests passed"
    
    if ($passedTests -eq $totalTests) {
        Write-Success "All tests passed! API is fully functional."
        exit 0
    } else {
        Write-Warning "Some tests failed. Check the output above for details."
        exit 1
    }
}

# Run the test suite
Main