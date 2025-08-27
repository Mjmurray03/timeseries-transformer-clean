# Time-Series Transformer API Reference

## Overview

The Time-Series Transformer API provides production-ready endpoints for stock price prediction using state-of-the-art transformer models. The API supports single and batch predictions with comprehensive error handling, authentication, rate limiting, caching, and monitoring.

## Base URL

```
http://localhost:8000  # Development
https://api.example.com  # Production
```

## Authentication

All prediction endpoints require authentication using API keys:

```http
Authorization: Bearer your_api_key_here
```

## Rate Limiting

- **Default Limit**: 100 requests per minute per API key
- **Batch Limit**: 100 items per batch request
- **Response Headers**:
  - `X-RateLimit-Limit`: Request limit per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Window reset timestamp

## Endpoints

### 🎯 Predictions

#### POST /predict

Generate prediction for a single ticker.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "features": [
    [100.5, 101.2, 99.8, 100.1, 1000000, 0.02, 0.5],
    [100.1, 102.0, 100.0, 101.5, 1200000, 0.014, 0.45]
    // ... 60 time steps total with 7 features each
  ],
  "horizon": 5  // Optional, default: 5, range: 1-30
}
```

**Feature Format:**
Each time step contains 7 features:
1. **Open Price** - Opening price for the period
2. **High Price** - Highest price during the period  
3. **Low Price** - Lowest price during the period
4. **Close Price** - Closing price for the period
5. **Volume** - Trading volume
6. **Return** - Price return (close/previous_close - 1)
7. **Volatility** - Price volatility measure

**Response:**
```json
{
  "prediction": [103.2, 103.8, 104.1, 104.5, 105.0],
  "confidence_intervals": {
    "68%": {
      "lower": [102.5, 103.1, 103.4, 103.8, 104.3],
      "upper": [103.9, 104.5, 104.8, 105.2, 105.7],
      "confidence_level": 0.68
    },
    "95%": {
      "lower": [101.8, 102.4, 102.7, 103.1, 103.6],
      "upper": [104.6, 105.2, 105.5, 105.9, 106.4],
      "confidence_level": 0.95
    }
  },
  "attention_weights": [
    [0.1, 0.15, 0.2, 0.18, 0.12, 0.15, 0.1]
    // Attention weights for each feature at each time step
  ],
  "metadata": {
    "model_version": "v1.0.0",
    "inference_time_ms": 45.2,
    "timestamp": "2024-01-15T10:30:00Z",
    "cache_hit": false,
    "request_id": "req_12345"
  }
}
```

**Status Codes:**
- `200` - Success
- `400` - Validation error
- `401` - Authentication failed
- `422` - Invalid input data
- `429` - Rate limit exceeded
- `500` - Internal server error

---

#### POST /batch_predict

Generate predictions for multiple tickers in a single request.

**Request Body:**
```json
{
  "requests": [
    {
      "ticker": "AAPL",
      "features": [[100.0] * 7] * 60,
      "horizon": 5
    },
    {
      "ticker": "GOOGL",
      "features": [[150.0] * 7] * 60,
      "horizon": 3
    }
    // Up to 100 requests per batch
  ]
}
```

**Response:**
```json
{
  "predictions": [
    // Array of PredictionResponse objects
  ],
  "batch_metadata": {
    "batch_id": "batch_67890",
    "total_requests": 2,
    "successful_predictions": 2,
    "failed_predictions": 0,
    "batch_processing_time_ms": 125.7,
    "timestamp": "2024-01-15T10:30:00Z",
    "errors": null
  }
}
```

**Performance:**
- **Target Latency**: < 500ms for batches up to 100 items
- **Parallel Processing**: Requests processed concurrently
- **Error Isolation**: Individual request failures don't affect the batch

---

### 🏥 Health & Monitoring

#### GET /health

Comprehensive health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "model_status": {
    "model_primary": "healthy"
  },
  "dependencies": {
    "redis": "healthy",
    "prediction_cache": "healthy"
  }
}
```

**Status Values:**
- `healthy` - All systems operational
- `unhealthy` - Critical systems down
- `degraded` - Some non-critical systems down

---

#### GET /ready

Kubernetes readiness probe endpoint.

**Response:**
```json
{
  "status": "ready"
}
```

**Status Codes:**
- `200` - Service ready to accept traffic
- `503` - Service not ready (still starting up)

---

#### GET /model_info

Get detailed model information.

**Response:**
```json
{
  "model_version": "v1.0.0",
  "architecture": "transformer",
  "parameters": 12500000,
  "device": "cuda:0",
  "loaded_at": "2024-01-15T09:00:00Z",
  "training_metrics": {
    "final_loss": 0.0023,
    "validation_rmse": 0.85,
    "validation_mae": 0.67
  }
}
```

---

#### GET /metrics

Prometheus metrics endpoint for monitoring and alerting.

**Response Format:** Prometheus exposition format

**Key Metrics:**
- `api_requests_total` - Total API requests by method/endpoint/status
- `api_request_duration_seconds` - Request duration histogram
- `model_inference_duration_seconds` - Model inference time
- `api_active_connections` - Current active connections
- `cache_hits_total` / `cache_misses_total` - Cache performance

---

#### GET /metrics/summary

Human-readable metrics summary.

**Response:**
```json
{
  "active_connections": 45,
  "total_requests": 12500,
  "cache_hit_rate": 85.2,
  "avg_inference_time_ms": 47.3,
  "error_rate": 0.8
}
```

---

### 🔄 WebSocket Streaming

#### WS /ws/stream/{ticker}

Real-time prediction streaming via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream/AAPL');
```

**Client Message:**
```json
{
  "type": "start_stream",
  "data": {
    "ticker": "AAPL",
    "update_interval": 5,
    "features_source": "live"
  }
}
```

**Server Messages:**
```json
{
  "type": "prediction",
  "data": {
    "ticker": "AAPL",
    "prediction": [103.2, 103.8, 104.1, 104.5, 105.0],
    "timestamp": "2024-01-15T10:30:00Z",
    "confidence": 0.95
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "ws_12345"
}
```

**Message Types:**
- `status` - Connection status updates
- `prediction` - Real-time predictions
- `error` - Error notifications
- `keepalive` - Connection keepalive pings

---

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "error_code": "VALIDATION_ERROR",
    "message": "Input validation failed",
    "details": {
      "validation_errors": [
        {
          "field": "features",
          "message": "Features must have exactly 60 time steps",
          "type": "value_error",
          "input": "provided_input"
        }
      ]
    },
    "request_id": "req_error_123"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Input validation failed |
| `AUTHENTICATION_ERROR` | 401 | Invalid or missing API key |
| `AUTHORIZATION_ERROR` | 403 | Access denied |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_ERROR` | 429 | Rate limit exceeded |
| `INFERENCE_ERROR` | 422 | Model inference failed |
| `TIMEOUT_ERROR` | 408 | Request timeout |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error |

### Retry Logic

Implement exponential backoff for retries:

```python
import asyncio
import random

async def retry_request(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
```

---

## Input Validation

### Ticker Symbol Requirements
- **Format**: 1-5 uppercase letters
- **Pattern**: `^[A-Z]{1,5}$`
- **Examples**: `AAPL`, `GOOGL`, `BRK.A`

### Features Requirements
- **Shape**: Exactly 60 time steps × 7 features
- **Data Type**: Floating point numbers
- **Range**: Features should be normalized (-10 to +10)
- **No NaN/Inf**: All values must be finite numbers

### Horizon Requirements
- **Range**: 1 to 30 days
- **Default**: 5 days
- **Type**: Integer

---

## Performance Characteristics

### Latency Targets
- **Single Prediction**: < 100ms (P95)
- **Batch Prediction**: < 500ms for 100 items (P95)
- **Health Check**: < 10ms

### Throughput Capacity
- **Single Requests**: 1,000 requests/second
- **Batch Requests**: 100 batches/second (10,000 predictions/second)
- **Concurrent Connections**: 1,000+

### Resource Usage
- **Memory**: ~4GB per model instance
- **CPU**: 2-4 cores recommended
- **GPU**: Optional, 1 GPU per model instance
- **Storage**: Minimal (models cached in memory)

---

## Caching Strategy

### Prediction Cache
- **Backend**: Redis
- **TTL**: 5 minutes (300 seconds)
- **Key Format**: SHA-256 hash of request content
- **Cache Headers**: `X-Cache-Status` in response

### Model Cache
- **Type**: In-memory cache
- **Persistence**: Until restart
- **Eviction**: LRU (Least Recently Used)
- **Size**: Configurable (default: 10 models)

---

## Security Features

### API Key Authentication
- **Format**: Bearer token
- **Length**: Minimum 32 characters
- **Validation**: Real-time against secure store
- **Caching**: 1-hour cache for valid keys

### Rate Limiting
- **Algorithm**: Token bucket
- **Granularity**: Per API key and per IP
- **Storage**: Redis-backed with memory fallback
- **Headers**: Rate limit information in response

### Security Headers
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy: default-src 'self'`

---

## Client Libraries

### Python Client

```python
from src.api.docs.api_examples import APIClient

client = APIClient(
    base_url="http://localhost:8000",
    api_key="your_api_key_here"
)

# Single prediction
result = await client.predict_single("AAPL", features, horizon=5)

# Batch prediction
results = await client.predict_batch(requests)

await client.close()
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class APIClient {
  constructor(baseUrl, apiKey) {
    this.client = axios.create({
      baseURL: baseUrl,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });
  }
  
  async predictSingle(ticker, features, horizon = 5) {
    const response = await this.client.post('/predict', {
      ticker, features, horizon
    });
    return response.data;
  }
}
```

### cURL Examples

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_api_key_here" \
     -d '{"ticker":"AAPL","features":[[100.0]*7]*60,"horizon":5}'

# Health check
curl -X GET "http://localhost:8000/health"

# Metrics
curl -X GET "http://localhost:8000/metrics"
```

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/api ./api
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: timeseries-transformer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: timeseries-transformer-api
  template:
    metadata:
      labels:
        app: timeseries-transformer-api
    spec:
      containers:
      - name: api
        image: timeseries-transformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/best_model.pt"
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi" 
            cpu: "4"
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

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model file | `models/best_model.pt` |
| `SCALER_PATH` | Path to scaler file | `models/scaler.pkl` |
| `DEVICE` | Compute device | `auto` |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_DB` | Redis database | `0` |
| `MODEL_POOL_SIZE` | Number of model instances | `1` |
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `WORKERS` | Uvicorn workers | `1` |

---

## Monitoring & Observability

### Metrics Collection

The API exposes Prometheus metrics for comprehensive monitoring:

```python
# Example Grafana queries
rate(api_requests_total[5m])  # Request rate
histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))  # P95 latency
api_active_connections  # Active connections
cache_hit_rate * 100  # Cache hit rate percentage
```

### Logging

Structured JSON logging with request tracing:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "req_12345",
  "method": "POST",
  "url": "/predict",
  "status_code": 200,
  "process_time_ms": 45.2,
  "user_id": "user_789"
}
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: api_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.01
    for: 2m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    annotations:
      summary: "High latency detected"
```

---

## Best Practices

### API Usage
1. **Batch Requests**: Use batch endpoints for multiple predictions
2. **Caching**: Identical requests are cached for 5 minutes
3. **Error Handling**: Implement proper retry logic with exponential backoff
4. **Rate Limiting**: Monitor rate limit headers and implement client-side limiting
5. **Request IDs**: Use request IDs from responses for support and debugging

### Performance Optimization
1. **Connection Pooling**: Reuse HTTP connections
2. **Parallel Requests**: Use async/await for concurrent requests
3. **Feature Preprocessing**: Preprocess features client-side when possible
4. **Monitoring**: Monitor latency and error rates

### Security
1. **API Keys**: Store API keys securely, rotate regularly
2. **HTTPS**: Always use HTTPS in production
3. **Input Validation**: Validate inputs before sending requests
4. **Error Handling**: Don't expose sensitive information in error messages

---

For more examples and advanced usage, see the [API Examples](api_examples.py) file.