# PowerShell Script to Train Models for All Available Tickers
# PURPOSE: Automate training of individual models for each stock ticker
# INPUTS: Reads available tickers from data/raw/ directory
# OUTPUTS: Trained models in models/, logs in logs/, progress tracking
# VERIFICATION: Error handling, progress logging, summary report

param(
    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [float]$LearningRate = 0.001,
    [int]$SeqLen = 60,
    [int]$Horizon = 3,
    [switch]$UseWandB,
    [string[]]$SpecificTickers = @()
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Setup logging
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs\batch_training"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$summaryLog = "$logDir\batch_training_$timestamp.log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $logEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Level] $Message"
    Write-Host $logEntry -ForegroundColor $(if ($Level -eq "ERROR") {"Red"} elseif ($Level -eq "WARNING") {"Yellow"} else {"White"})
    Add-Content -Path $summaryLog -Value $logEntry
}

function Get-AvailableTickers {
    $dataDir = "data\raw"
    if (-not (Test-Path $dataDir)) {
        throw "Data directory not found: $dataDir"
    }
    
    $tickers = @()
    Get-ChildItem -Path $dataDir -Directory | ForEach-Object {
        $tickerDir = $_.FullName
        $parquetFiles = Get-ChildItem -Path $tickerDir -Filter "*.parquet" -ErrorAction SilentlyContinue
        if ($parquetFiles.Count -gt 0) {
            $tickers += $_.Name
        }
    }
    
    return $tickers | Sort-Object
}

function Test-GPUAvailable {
    try {
        $result = python -c "import torch; print(torch.cuda.is_available())" 2>$null
        return $result -eq "True"
    } catch {
        return $false
    }
}

function Get-GPUMemory {
    try {
        $memory = python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}' if torch.cuda.is_available() else '0')" 2>$null
        return $memory
    } catch {
        return "0"
    }
}

function Train-SingleTicker {
    param(
        [string]$Ticker,
        [hashtable]$Parameters
    )
    
    $startTime = Get-Date
    Write-Log "Starting training for $Ticker" "INFO"
    
    # Build command
    $cmd = "python scripts\training\train_ultra_simple.py"
    $cmd += " --ticker $Ticker"
    $cmd += " --epochs $($Parameters.Epochs)"
    $cmd += " --batch-size $($Parameters.BatchSize)"
    $cmd += " --learning-rate $($Parameters.LearningRate)"
    $cmd += " --seq-len $($Parameters.SeqLen)"
    $cmd += " --horizon $($Parameters.Horizon)"
    
    if ($Parameters.UseWandB) {
        $cmd += " --use-wandb"
    }
    
    # Execute training
    try {
        Write-Log "Executing: $cmd" "INFO"
        $output = Invoke-Expression $cmd 2>&1
        
        # Check if model was saved
        $modelPath = "models\model_${Ticker}_best.pt"
        if (Test-Path $modelPath) {
            $duration = (Get-Date) - $startTime
            Write-Log "Successfully trained $Ticker in $($duration.TotalMinutes.ToString('F2')) minutes" "INFO"
            
            # Extract best RMSE from output if available
            $rmseMatch = $output | Select-String -Pattern "Best Val RMSE: ([\d.]+)"
            if ($rmseMatch) {
                $rmse = $rmseMatch.Matches[0].Groups[1].Value
                Write-Log "Best RMSE for $Ticker`: $rmse" "INFO"
                return @{
                    Success = $true
                    Ticker = $Ticker
                    RMSE = $rmse
                    Duration = $duration
                    ModelPath = $modelPath
                }
            }
            
            return @{
                Success = $true
                Ticker = $Ticker
                Duration = $duration
                ModelPath = $modelPath
            }
        } else {
            throw "Model file not created"
        }
    } catch {
        Write-Log "Failed to train $Ticker`: $_" "ERROR"
        return @{
            Success = $false
            Ticker = $Ticker
            Error = $_.Exception.Message
        }
    }
}

# Main execution
Write-Log "="*60 "INFO"
Write-Log "Multi-Ticker Training Automation Script" "INFO"
Write-Log "="*60 "INFO"

# Check environment
Write-Log "Checking environment..." "INFO"
$gpuAvailable = Test-GPUAvailable
if ($gpuAvailable) {
    $gpuMemory = Get-GPUMemory
    Write-Log "GPU Available: Yes (Memory: $gpuMemory GB)" "INFO"
} else {
    Write-Log "GPU Available: No (Using CPU)" "WARNING"
}

# Get tickers to train
$availableTickers = Get-AvailableTickers
Write-Log "Found $($availableTickers.Count) tickers with data: $($availableTickers -join ', ')" "INFO"

# Filter tickers if specific ones requested
if ($SpecificTickers.Count -gt 0) {
    $tickersToTrain = $availableTickers | Where-Object { $_ -in $SpecificTickers }
    Write-Log "Training specific tickers: $($tickersToTrain -join ', ')" "INFO"
} else {
    $tickersToTrain = $availableTickers
    Write-Log "Training all available tickers" "INFO"
}

# Validate we have tickers to train
if ($tickersToTrain.Count -eq 0) {
    Write-Log "No valid tickers to train" "ERROR"
    exit 1
}

# Training parameters
$trainingParams = @{
    Epochs = $Epochs
    BatchSize = $BatchSize
    LearningRate = $LearningRate
    SeqLen = $SeqLen
    Horizon = $Horizon
    UseWandB = $UseWandB
}

Write-Log "Training parameters:" "INFO"
$trainingParams.GetEnumerator() | ForEach-Object {
    Write-Log "  $($_.Key): $($_.Value)" "INFO"
}

# Create required directories
@("models", "models\scalers", "logs") | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

# Train each ticker
$results = @()
$successCount = 0
$failCount = 0
$totalStartTime = Get-Date

Write-Log "`nStarting training loop..." "INFO"
Write-Log "-"*40 "INFO"

for ($i = 0; $i -lt $tickersToTrain.Count; $i++) {
    $ticker = $tickersToTrain[$i]
    $progress = ($i + 1).ToString() + "/" + $tickersToTrain.Count.ToString()
    
    Write-Log "`n[$progress] Training $ticker..." "INFO"
    
    $result = Train-SingleTicker -Ticker $ticker -Parameters $trainingParams
    $results += $result
    
    if ($result.Success) {
        $successCount++
        Write-Log "[$progress] $ticker completed successfully" "INFO"
    } else {
        $failCount++
        Write-Log "[$progress] $ticker failed" "ERROR"
    }
    
    # Show progress
    $percentComplete = [math]::Round(($i + 1) / $tickersToTrain.Count * 100, 1)
    Write-Progress -Activity "Training Models" -Status "$ticker ($progress)" -PercentComplete $percentComplete
}

Write-Progress -Activity "Training Models" -Completed

# Calculate total duration
$totalDuration = (Get-Date) - $totalStartTime

# Generate summary report
Write-Log "`n" + "="*60 "INFO"
Write-Log "TRAINING SUMMARY" "INFO"
Write-Log "="*60 "INFO"
Write-Log "Total Tickers: $($tickersToTrain.Count)" "INFO"
Write-Log "Successful: $successCount" "INFO"
Write-Log "Failed: $failCount" "INFO"
Write-Log "Total Duration: $($totalDuration.TotalMinutes.ToString('F2')) minutes" "INFO"
Write-Log "Average Time per Ticker: $([math]::Round($totalDuration.TotalMinutes / $tickersToTrain.Count, 2)) minutes" "INFO"

# Successful models summary
if ($successCount -gt 0) {
    Write-Log "`nSuccessful Models:" "INFO"
    $results | Where-Object { $_.Success } | ForEach-Object {
        $msg = "  - $($_.Ticker)"
        if ($_.RMSE) {
            $msg += " (RMSE: $($_.RMSE))"
        }
        if ($_.Duration) {
            $msg += " [Time: $($_.Duration.TotalMinutes.ToString('F2')) min]"
        }
        Write-Log $msg "INFO"
    }
}

# Failed models summary
if ($failCount -gt 0) {
    Write-Log "`nFailed Models:" "ERROR"
    $results | Where-Object { -not $_.Success } | ForEach-Object {
        Write-Log "  - $($_.Ticker): $($_.Error)" "ERROR"
    }
}

# Save detailed results to JSON
$resultsFile = "$logDir\training_results_$timestamp.json"
$results | ConvertTo-Json -Depth 3 | Out-File -FilePath $resultsFile
Write-Log "`nDetailed results saved to: $resultsFile" "INFO"

# Model verification
Write-Log "`nVerifying saved models..." "INFO"
$modelCount = (Get-ChildItem -Path "models" -Filter "model_*_best.pt" -ErrorAction SilentlyContinue).Count
$scalerCount = (Get-ChildItem -Path "models\scalers" -Filter "scaler_*.json" -ErrorAction SilentlyContinue).Count
Write-Log "Models found: $modelCount" "INFO"
Write-Log "Scalers found: $scalerCount" "INFO"

# Exit with appropriate code
if ($failCount -eq 0) {
    Write-Log "`nAll training completed successfully!" "INFO"
    exit 0
} elseif ($successCount -gt 0) {
    Write-Log "`nPartial success: $successCount/$($tickersToTrain.Count) models trained" "WARNING"
    exit 2
} else {
    Write-Log "`nAll training failed!" "ERROR"
    exit 1
}