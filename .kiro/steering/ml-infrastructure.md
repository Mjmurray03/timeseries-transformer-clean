# ML Infrastructure Standards
---
inclusion: always
priority: 2
---

## Model Versioning Strategy

### Version Nomenclature
```
v{major}.{minor}.{patch}-{variant}
Examples:
- v1.0.0-base: Initial transformer model
- v1.1.0-large: Scaled architecture
- v1.1.1-fix: Bug fixes
- v2.0.0-ensemble: Major architecture change
```

### Model Registry Structure
```python
MODEL_REGISTRY = {
    "model_id": "timeseries-transformer",
    "version": "1.0.0",
    "architecture": "transformer-6layer",
    "training_data": "SP500_2019-2024",
    "metrics": {
        "val_rmse": 0.0187,
        "val_sharpe": 1.23,
        "val_accuracy": 0.545
    },
    "artifacts": {
        "model_weights": "s3://models/v1.0.0/weights.pt",
        "config": "s3://models/v1.0.0/config.json",
        "scalers": "s3://models/v1.0.0/scalers.pkl"
    }
}
```

## Training Pipeline Patterns

### Data Pipeline
```python
# Always use deterministic operations
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Data loading pattern
def create_data_pipeline(config):
    # 1. Download with retry logic
    raw_data = download_with_retry(config.tickers, max_retries=3)
    
    # 2. Validate data quality
    validated_data = validate_completeness(raw_data, min_days=252)
    
    # 3. Feature engineering
    features = engineer_features(validated_data, config.indicators)
    
    # 4. Sequence generation
    sequences = build_sequences(features, window=60, horizon=5)
    
    # 5. Train/val/test split (time-aware)
    splits = time_series_split(sequences, ratios=[0.7, 0.15, 0.15])
    
    return splits
```

### Training Loop Pattern
```python
def training_loop(model, train_loader, val_loader, config):
    # Setup
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler()  # Mixed precision
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        for batch in train_loader:
            with autocast():  # FP16
                loss = train_step(model, batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        val_metrics = validate(model, val_loader)
        
        # Checkpointing
        if val_metrics['loss'] < best_loss:
            save_checkpoint(model, optimizer, epoch, val_metrics)
        
        # Early stopping
        if early_stopping.should_stop(val_metrics['loss']):
            break
        
        scheduler.step()
```

## GPU Resource Management

### Memory Optimization Strategies
```python
# 1. Gradient accumulation for large effective batch sizes
ACCUMULATION_STEPS = 4
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / ACCUMULATION_STEPS
    loss.backward()
    
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Gradient checkpointing for deep models
class CheckpointedTransformer(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x)  # Trade compute for memory
        return x

# 3. Mixed precision training
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# 4. Efficient data loading
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### Multi-GPU Patterns (Future)
```python
# DataParallel (simple but less efficient)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# DistributedDataParallel (recommended)
torch.distributed.init_process_group(backend='nccl')
model = DDP(model, device_ids=[local_rank])
```

## Experiment Tracking

### W&B Integration Pattern
```python
import wandb

wandb.init(
    project="timeseries-transformer",
    config={
        "architecture": "transformer",
        "dataset": "SP500",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-4
    },
    tags=["production", "v1.0.0"]
)

# Log metrics
wandb.log({
    "train/loss": train_loss,
    "val/rmse": val_rmse,
    "val/sharpe": sharpe_ratio,
    "learning_rate": scheduler.get_last_lr()[0]
})

# Log artifacts
wandb.save("model_checkpoint.pt")
wandb.save("predictions.png")
```

### Experiment Naming Convention
```
{model_type}_{dataset}_{key_hyperparam}_{timestamp}
Examples:
- transformer_sp500_lr1e4_20240315
- lstm_tech_stocks_layers3_20240316
- ensemble_all_stocks_models5_20240317
```

## Model Deployment Patterns

### Model Serving Architecture
```python
class ModelServer:
    def __init__(self, model_path, device='cuda'):
        self.model = self.load_model(model_path)
        self.model.eval()
        self.device = device
        
        # Warmup
        self.warmup()
        
    @torch.no_grad()
    def predict(self, data):
        # Preprocess
        tensor = self.preprocess(data)
        
        # Inference with TorchScript
        output = self.model(tensor)
        
        # Postprocess
        prediction = self.postprocess(output)
        
        return prediction
    
    def warmup(self):
        # Run dummy predictions to initialize CUDA
        dummy = torch.randn(1, 60, 7).to(self.device)
        _ = self.model(dummy)
```

### A/B Testing Framework
```python
class ABTestManager:
    def __init__(self):
        self.models = {
            'control': load_model('v1.0.0'),
            'treatment': load_model('v1.1.0')
        }
        self.traffic_split = 0.1  # 10% to treatment
        
    def route_request(self, request_id):
        if hash(request_id) % 100 < self.traffic_split * 100:
            return 'treatment'
        return 'control'
    
    def predict(self, data, request_id):
        model_version = self.route_request(request_id)
        model = self.models[model_version]
        
        # Log for analysis
        log_prediction(request_id, model_version, data)
        
        return model.predict(data)
```

## Performance Monitoring

### Key Metrics to Track
```python
PRODUCTION_METRICS = {
    # Model performance
    "prediction_rmse": RollingWindow(window=1000),
    "directional_accuracy": RollingWindow(window=1000),
    "prediction_drift": KSDrift(reference_data),
    
    # System performance
    "inference_latency_p50": Histogram(),
    "inference_latency_p99": Histogram(),
    "gpu_memory_usage": Gauge(),
    "batch_throughput": Rate(),
    
    # Business metrics
    "daily_sharpe_ratio": DailyMetric(),
    "cumulative_returns": CumulativeSum(),
    "max_drawdown": MaxDrawdown(),
    
    # Data quality
    "missing_features": Counter(),
    "outlier_inputs": Counter(),
    "feature_drift": DataDriftDetector()
}
```

### Alert Thresholds
```yaml
alerts:
  - name: high_inference_latency
    condition: p99_latency > 100ms
    severity: warning
    
  - name: model_performance_degradation
    condition: rolling_rmse > baseline_rmse * 1.2
    severity: critical
    
  - name: gpu_memory_critical
    condition: gpu_memory_usage > 22GB
    severity: critical
    
  - name: prediction_drift_detected
    condition: ks_statistic > 0.1
    severity: warning
```

## Continuous Learning Pipeline

### Retraining Triggers
1. **Scheduled**: Weekly retrain with latest data
2. **Performance-based**: RMSE degrades >20% from baseline
3. **Drift-based**: Feature or prediction distribution shift detected
4. **Market-event**: Major market regime change (VIX spike, etc.)

### Retraining Workflow
```python
def automated_retrain():
    # 1. Collect recent data
    new_data = collect_recent_data(days=30)
    
    # 2. Validate data quality
    if not validate_data_quality(new_data):
        alert("Data quality check failed")
        return
    
    # 3. Retrain model
    updated_model = retrain_model(
        base_model=current_model,
        new_data=new_data,
        fine_tune_only=True  # Freeze early layers
    )
    
    # 4. Validate performance
    metrics = evaluate_model(updated_model, test_data)
    
    # 5. Deploy if improved
    if metrics['rmse'] < current_metrics['rmse'] * 0.95:
        deploy_model(updated_model, canary_ratio=0.1)
    else:
        log("Retrained model didn't improve, keeping current")
```

## Security Considerations

### Model Security
- Store model weights encrypted at rest
- Use signed model files to prevent tampering
- Implement model access logging
- Regular security audits of inference endpoints

### Data Security
- PII removal from training data
- Differential privacy for sensitive features
- Secure data transmission (TLS 1.3)
- API key rotation every 30 days

### Inference Security
```python
class SecureModelServer:
    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=100, window=60)
        self.input_validator = InputValidator()
        
    def predict(self, request):
        # Rate limiting
        if not self.rate_limiter.allow(request.client_id):
            raise RateLimitExceeded()
        
        # Input validation
        if not self.input_validator.validate(request.data):
            raise InvalidInput()
        
        # Audit logging
        audit_log.record(request)
        
        # Actual prediction
        return self.model.predict(request.data)
```