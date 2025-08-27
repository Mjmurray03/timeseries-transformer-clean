# Deployment Standards
---
inclusion: fileMatch
fileMatchPattern: "deployment/**/*"
priority: 2
---

## Container Specifications

### Docker Base Images
```dockerfile
# Training Image - Full ML stack
FROM nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04 AS training
WORKDIR /app
RUN apt-get update && apt-get install -y python3.10 python3-pip
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Inference Image - Minimal footprint
FROM python:3.10-slim AS inference
WORKDIR /app
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt
COPY --from=training /app/models /app/models
```

### Multi-stage Build Pattern
```dockerfile
# Stage 1: Builder
FROM python:3.10 AS builder
WORKDIR /build
COPY . .
RUN pip install poetry && poetry build

# Stage 2: Testing
FROM builder AS tester
RUN poetry install --with dev
RUN pytest tests/ --cov=src --cov-report=xml

# Stage 3: Production
FROM python:3.10-slim AS production
WORKDIR /app
COPY --from=builder /build/dist/*.whl .
RUN pip install *.whl && rm *.whl
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Kubernetes Deployment Patterns

### Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: timeseries-transformer
  labels:
    app: timeseries-transformer
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: timeseries-transformer
  template:
    metadata:
      labels:
        app: timeseries-transformer
        version: v1.0.0
    spec:
      containers:
      - name: model-server
        image: timeseries-transformer:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: "1"  # GPU request
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: "1"
        env:
        - name: MODEL_VERSION
          value: "v1.0.0"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: timeseries-transformer-service
spec:
  selector:
    app: timeseries-transformer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: LoadBalancer
  sessionAffinity: ClientIP  # Sticky sessions for consistent predictions
```

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: timeseries-transformer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: timeseries-transformer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_latency_p99
      target:
        type: AverageValue
        averageValue: "100m"  # 100ms
```

## API Gateway Configuration

### NGINX Configuration
```nginx
upstream model_backend {
    least_conn;  # Load balancing strategy
    server model-server-1:8000 weight=3 max_fails=3 fail_timeout=30s;
    server model-server-2:8000 weight=3 max_fails=3 fail_timeout=30s;
    server model-server-3:8000 weight=1 max_fails=3 fail_timeout=30s;  # Canary
}

server {
    listen 443 ssl http2;
    server_name api.timeseries-transformer.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;
    ssl_protocols TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
    
    # Caching
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=model_cache:10m max_size=1g inactive=60m;
    proxy_cache_key "$request_method$request_uri$request_body";
    
    location /predict {
        proxy_pass http://model_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Caching for identical requests
        proxy_cache model_cache;
        proxy_cache_valid 200 5m;
        proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 30s;
    }
    
    location /health {
        proxy_pass http://model_backend/health;
        access_log off;  # Don't log health checks
    }
}
```

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: Deploy ML Model

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install
    
    - name: Run tests
      run: |
        poetry run pytest tests/ --cov=src --cov-report=xml
        poetry run mypy src/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/timeseries-transformer:latest
          ${{ secrets.DOCKER_USERNAME }}/timeseries-transformer:${{ github.ref_name }}
        cache-from: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/timeseries-transformer:buildcache
        cache-to: type=registry,ref=${{ secrets.DOCKER_USERNAME }}/timeseries-transformer:buildcache,mode=max
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
    - name: Deploy to Kubernetes
      env:
        KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
      run: |
        echo "$KUBE_CONFIG" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        
        # Update deployment
        kubectl set image deployment/timeseries-transformer \
          model-server=${{ secrets.DOCKER_USERNAME }}/timeseries-transformer:${{ github.ref_name }}
        
        # Wait for rollout
        kubectl rollout status deployment/timeseries-transformer
        
        # Run smoke tests
        kubectl exec -it deployment/timeseries-transformer -- python -m pytest tests/smoke
```

## Deployment Checklist

### Pre-deployment Validation
```python
class DeploymentValidator:
    """Validate model before deployment"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.checks = []
    
    def validate(self) -> bool:
        """Run all validation checks"""
        self.checks = [
            self.check_model_size(),
            self.check_inference_speed(),
            self.check_memory_usage(),
            self.check_prediction_quality(),
            self.check_api_compatibility()
        ]
        return all(self.checks)
    
    def check_model_size(self) -> bool:
        """Ensure model fits in container"""
        size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        return size_mb < 100  # Max 100MB
    
    def check_inference_speed(self) -> bool:
        """Ensure inference meets SLA"""
        model = torch.jit.load(self.model_path)
        dummy_input = torch.randn(1, 60, 7)
        
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Time inference
        times = []
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            times.append(time.time() - start)
        
        p99_latency = np.percentile(times, 99)
        return p99_latency < 0.1  # 100ms SLA
    
    def check_memory_usage(self) -> bool:
        """Ensure model fits in memory"""
        import tracemalloc
        
        tracemalloc.start()
        model = torch.jit.load(self.model_path)
        _ = model(torch.randn(32, 60, 7))  # Batch inference
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return peak / (1024 * 1024) < 2000  # Max 2GB
    
    def check_prediction_quality(self) -> bool:
        """Ensure predictions are reasonable"""
        model = torch.jit.load(self.model_path)
        
        # Test data
        test_input = torch.randn(10, 60, 7)
        predictions = model(test_input)
        
        # Check predictions are in reasonable range
        return predictions.std() > 0.01 and predictions.std() < 10
    
    def check_api_compatibility(self) -> bool:
        """Ensure model works with API"""
        try:
            # Simulate API request
            from src.api.schemas import PredictionRequest
            from src.inference.predictor import ModelPredictor
            
            predictor = ModelPredictor(self.model_path)
            request = PredictionRequest(
                ticker="AAPL",
                features=[[1.0] * 7] * 60
            )
            response = predictor.predict(request)
            
            return response is not None
        except Exception as e:
            logger.error(f"API compatibility check failed: {e}")
            return False
```

## Rollback Strategy

### Automated Rollback
```python
class RollbackManager:
    """Manage deployment rollbacks"""
    
    def __init__(self, metrics_client):
        self.metrics = metrics_client
        self.baseline_metrics = {}
    
    def should_rollback(self, deployment_id: str) -> bool:
        """Determine if rollback is needed"""
        current_metrics = self.metrics.get_current_metrics()
        baseline = self.baseline_metrics.get(deployment_id)
        
        if not baseline:
            return False
        
        # Check critical metrics
        checks = [
            current_metrics['error_rate'] > baseline['error_rate'] * 1.5,
            current_metrics['p99_latency'] > baseline['p99_latency'] * 1.5,
            current_metrics['rmse'] > baseline['rmse'] * 1.2,
            current_metrics['availability'] < 0.99
        ]
        
        return any(checks)
    
    def execute_rollback(self, deployment_id: str):
        """Execute rollback to previous version"""
        logger.warning(f"Initiating rollback for {deployment_id}")
        
        # Kubernetes rollback
        subprocess.run([
            "kubectl", "rollout", "undo",
            "deployment/timeseries-transformer"
        ])
        
        # Wait for rollback
        subprocess.run([
            "kubectl", "rollout", "status",
            "deployment/timeseries-transformer"
        ])
        
        # Alert team
        self.send_rollback_alert(deployment_id)
```

## Monitoring Configuration

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'ticker']
)

prediction_latency = Histogram(
    'model_prediction_duration_seconds',
    'Prediction latency in seconds',
    ['model_version'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_version', 'metric_type']
)

gpu_memory_usage = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['device']
)

# Instrument code
@prediction_latency.time()
def predict(data):
    prediction_counter.labels(model_version='v1.0.0', ticker=data.ticker).inc()
    # Actual prediction logic
    return model.predict(data)
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Time-Series Transformer Monitoring",
    "panels": [
      {
        "title": "Prediction Rate",
        "targets": [
          {
            "expr": "rate(model_predictions_total[5m])",
            "legendFormat": "{{model_version}}"
          }
        ]
      },
      {
        "title": "Latency P99",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(model_prediction_duration_seconds_bucket[5m]))",
            "legendFormat": "P99 Latency"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "targets": [
          {
            "expr": "model_accuracy{metric_type='rmse'}",
            "legendFormat": "RMSE"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "gpu_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "GPU Memory (GB)"
          }
        ]
      }
    ]
  }
}
```