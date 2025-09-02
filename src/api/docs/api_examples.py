"""
API Usage Examples and Documentation

This module contains comprehensive examples for using the Time-Series Transformer API,
including request/response examples, error handling, and best practices.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

import httpx


class APIClient:
    """Example API client implementation"""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def predict_single(
        self, ticker: str, features: List[List[float]], horizon: int = 5
    ) -> Dict[str, Any]:
        """
        Make a single prediction request

        Example usage:
        ```python
        client = APIClient(api_key="your_api_key_here")

        # Prepare 60 time steps with 7 features each
        features = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] for _ in range(60)]

        result = await client.predict_single("AAPL", features, horizon=5)
        print(f"Prediction: {result['prediction']}")
        ```
        """
        payload = {"ticker": ticker, "features": features, "horizon": horizon}

        response = await self.client.post(
            f"{self.base_url}/predict", json=payload, headers=self._get_headers()
        )

        if response.status_code != 200:
            error_data = response.json()
            raise Exception(f"API Error: {error_data}")

        return response.json()

    async def predict_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a batch prediction request

        Example usage:
        ```python
        requests = [
            {
                "ticker": "AAPL",
                "features": [[1.0] * 7] * 60,
                "horizon": 5
            },
            {
                "ticker": "GOOGL",
                "features": [[2.0] * 7] * 60,
                "horizon": 3
            }
        ]

        result = await client.predict_batch(requests)
        print(f"Batch predictions: {len(result['predictions'])}")
        ```
        """
        payload = {"requests": requests}

        response = await self.client.post(
            f"{self.base_url}/batch_predict", json=payload, headers=self._get_headers()
        )

        if response.status_code != 200:
            error_data = response.json()
            raise Exception(f"API Error: {error_data}")

        return response.json()

    async def get_health(self) -> Dict[str, Any]:
        """Get API health status"""
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = await self.client.get(f"{self.base_url}/model_info", headers=self._get_headers())
        return response.json()

    async def get_metrics(self) -> Dict[str, Any]:
        """Get API metrics"""
        response = await self.client.get(f"{self.base_url}/metrics/summary")
        return response.json()

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Example request/response data
EXAMPLE_REQUESTS = {
    "single_prediction": {
        "ticker": "AAPL",
        "features": [
            [
                100.5,
                101.2,
                99.8,
                100.1,
                1000000,
                0.02,
                0.5,
            ],  # Day 1: [open, high, low, close, volume, return, volatility]
            [100.1, 102.0, 100.0, 101.5, 1200000, 0.014, 0.45],  # Day 2
            [101.5, 103.2, 101.0, 102.8, 1100000, 0.013, 0.48],  # Day 3
            # ... 57 more time steps (total 60)
        ]
        + [
            [
                102.0 + i * 0.1,
                103.0 + i * 0.1,
                101.0 + i * 0.1,
                102.5 + i * 0.1,
                1000000 + i * 10000,
                0.01 + i * 0.001,
                0.5 + i * 0.01,
            ]
            for i in range(57)
        ],
        "horizon": 5,
    },
    "batch_prediction": {
        "requests": [
            {
                "ticker": "AAPL",
                "features": [[100.0 + i * 0.1] * 7 for i in range(60)],
                "horizon": 5,
            },
            {
                "ticker": "GOOGL",
                "features": [[150.0 + i * 0.2] * 7 for i in range(60)],
                "horizon": 3,
            },
            {
                "ticker": "MSFT",
                "features": [[200.0 + i * 0.15] * 7 for i in range(60)],
                "horizon": 7,
            },
        ]
    },
}

EXAMPLE_RESPONSES = {
    "single_prediction": {
        "prediction": [103.2, 103.8, 104.1, 104.5, 105.0],
        "confidence_intervals": {
            "68%": {
                "lower": [102.5, 103.1, 103.4, 103.8, 104.3],
                "upper": [103.9, 104.5, 104.8, 105.2, 105.7],
                "confidence_level": 0.68,
            },
            "95%": {
                "lower": [101.8, 102.4, 102.7, 103.1, 103.6],
                "upper": [104.6, 105.2, 105.5, 105.9, 106.4],
                "confidence_level": 0.95,
            },
        },
        "attention_weights": [
            [0.1, 0.15, 0.2, 0.18, 0.12, 0.15, 0.1],  # Attention for each feature at each time step
            # ... more attention weights for each time step
        ],
        "metadata": {
            "model_version": "v1.0.0",
            "inference_time_ms": 45.2,
            "timestamp": "2024-01-15T10:30:00Z",
            "cache_hit": False,
            "request_id": "req_12345",
        },
    },
    "batch_prediction": {
        "predictions": [
            # Individual prediction responses for each request
        ],
        "batch_metadata": {
            "batch_id": "batch_67890",
            "total_requests": 3,
            "successful_predictions": 3,
            "failed_predictions": 0,
            "batch_processing_time_ms": 125.7,
            "timestamp": "2024-01-15T10:30:00Z",
            "errors": None,
        },
    },
    "health_check": {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z",
        "version": "1.0.0",
        "uptime_seconds": 86400,
        "model_status": {"model_primary": "healthy"},
        "dependencies": {"redis": "healthy", "prediction_cache": "healthy"},
    },
    "model_info": {
        "model_version": "v1.0.0",
        "architecture": "transformer",
        "parameters": 12500000,
        "device": "cuda:0",
        "loaded_at": "2024-01-15T09:00:00Z",
        "training_metrics": {"final_loss": 0.0023, "validation_rmse": 0.85, "validation_mae": 0.67},
    },
}

ERROR_EXAMPLES = {
    "validation_error": {
        "error": {
            "error_code": "VALIDATION_ERROR",
            "message": "Input validation failed",
            "details": {
                "validation_errors": [
                    {
                        "field": "features",
                        "message": "Features must have exactly 60 time steps",
                        "type": "value_error",
                        "input": "[[1.0, 2.0]]",  # Only 1 time step provided
                    }
                ]
            },
            "request_id": "req_error_123",
        },
        "timestamp": "2024-01-15T10:30:00Z",
    },
    "rate_limit_error": {
        "error": {
            "error_code": "RATE_LIMIT_ERROR",
            "message": "Rate limit exceeded",
            "details": {"retry_after_seconds": 60},
            "request_id": "req_rate_limit_456",
        },
        "timestamp": "2024-01-15T10:30:00Z",
    },
    "authentication_error": {
        "error": {
            "error_code": "AUTHENTICATION_ERROR",
            "message": "Invalid API key",
            "request_id": "req_auth_789",
        },
        "timestamp": "2024-01-15T10:30:00Z",
    },
}

CURL_EXAMPLES = {
    "single_prediction": """
# Single Prediction Request
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer your_api_key_here" \\
     -d '{
       "ticker": "AAPL",
       "features": [
         [100.5, 101.2, 99.8, 100.1, 1000000, 0.02, 0.5],
         [100.1, 102.0, 100.0, 101.5, 1200000, 0.014, 0.45]
       ],
       "horizon": 5
     }'
    """,
    "batch_prediction": """
# Batch Prediction Request  
curl -X POST "http://localhost:8000/batch_predict" \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer your_api_key_here" \\
     -d '{
       "requests": [
         {
           "ticker": "AAPL",
           "features": [[100.0, 101.0, 99.0, 100.5, 1000000, 0.01, 0.5]],
           "horizon": 5
         },
         {
           "ticker": "GOOGL", 
           "features": [[150.0, 151.0, 149.0, 150.5, 800000, 0.005, 0.4]],
           "horizon": 3
         }
       ]
     }'
    """,
    "health_check": """
# Health Check
curl -X GET "http://localhost:8000/health"
    """,
    "model_info": """
# Model Information
curl -X GET "http://localhost:8000/model_info" \\
     -H "Authorization: Bearer your_api_key_here"
    """,
}

PYTHON_EXAMPLES = {
    "basic_usage": '''
import asyncio
import httpx
from typing import List

async def predict_stock_price():
    """Basic usage example"""
    
    # Initialize client
    client = APIClient(
        base_url="http://localhost:8000",
        api_key="your_api_key_here"
    )
    
    try:
        # Prepare features (60 time steps with 7 features each)
        features = []
        for i in range(60):
            # Example: [open, high, low, close, volume, return, volatility]
            time_step = [
                100.0 + i * 0.1,  # open
                101.0 + i * 0.1,  # high  
                99.0 + i * 0.1,   # low
                100.5 + i * 0.1,  # close
                1000000 + i * 1000, # volume
                0.01 + i * 0.001, # return
                0.5 + i * 0.01    # volatility
            ]
            features.append(time_step)
        
        # Make prediction
        result = await client.predict_single(
            ticker="AAPL",
            features=features,
            horizon=5
        )
        
        print(f"Predicted prices: {result['prediction']}")
        print(f"Confidence intervals: {result['confidence_intervals']}")
        print(f"Processing time: {result['metadata']['inference_time_ms']}ms")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await client.close()

# Run the example
asyncio.run(predict_stock_price())
    ''',
    "batch_processing": '''
async def process_multiple_stocks():
    """Batch processing example"""
    
    client = APIClient(api_key="your_api_key_here")
    
    # Prepare multiple requests
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    requests = []
    
    for ticker in tickers:
        # Generate mock features for each ticker
        features = [[100.0 + i * 0.1] * 7 for i in range(60)]
        requests.append({
            "ticker": ticker,
            "features": features,
            "horizon": 5
        })
    
    try:
        # Make batch prediction
        result = await client.predict_batch(requests)
        
        print(f"Processed {result['batch_metadata']['successful_predictions']} predictions")
        print(f"Batch time: {result['batch_metadata']['batch_processing_time_ms']}ms")
        
        # Process individual predictions
        for i, prediction in enumerate(result['predictions']):
            ticker = tickers[i]
            predicted_prices = prediction['prediction']
            print(f"{ticker}: {predicted_prices}")
            
    except Exception as e:
        print(f"Batch processing error: {e}")
    
    finally:
        await client.close()

asyncio.run(process_multiple_stocks())
    ''',
    "error_handling": '''
async def handle_api_errors():
    """Error handling example"""
    
    client = APIClient(api_key="invalid_key")
    
    try:
        # This will fail with authentication error
        result = await client.predict_single(
            ticker="AAPL",
            features=[[1.0] * 7] * 60,  # Valid features
            horizon=5
        )
        
    except Exception as e:
        error_data = json.loads(str(e).replace("API Error: ", ""))
        
        error_code = error_data['error']['error_code']
        error_message = error_data['error']['message']
        request_id = error_data['error']['request_id']
        
        print(f"Error Code: {error_code}")
        print(f"Message: {error_message}")
        print(f"Request ID: {request_id}")
        
        # Handle specific error types
        if error_code == "AUTHENTICATION_ERROR":
            print("Please check your API key")
        elif error_code == "VALIDATION_ERROR":
            print("Please check your input data")
        elif error_code == "RATE_LIMIT_ERROR":
            retry_after = error_data['error']['details']['retry_after_seconds']
            print(f"Rate limited. Retry after {retry_after} seconds")
    
    finally:
        await client.close()

asyncio.run(handle_api_errors())
    ''',
}


def print_documentation():
    """Print comprehensive API documentation"""

    print("=" * 80)
    print("TIME-SERIES TRANSFORMER API DOCUMENTATION")
    print("=" * 80)

    print("\n📋 OVERVIEW")
    print("-" * 40)
    print("Production-ready API for stock price prediction using transformer models.")
    print("Features: Real-time predictions, batch processing, caching, monitoring.")

    print("\n🚀 QUICK START")
    print("-" * 40)
    print("1. Install dependencies: pip install -r requirements-api.txt")
    print("2. Start server: uvicorn src.api.main:app --reload")
    print("3. Access docs: http://localhost:8000/docs")

    print("\n📊 ENDPOINTS")
    print("-" * 40)
    print("POST /predict          - Single prediction")
    print("POST /batch_predict    - Batch predictions")
    print("GET  /health          - Health check")
    print("GET  /ready           - Readiness check")
    print("GET  /model_info      - Model information")
    print("GET  /metrics         - Prometheus metrics")
    print("GET  /metrics/summary - Metrics summary")
    print("WS   /ws/stream/{ticker} - WebSocket streaming")

    print("\n🔑 AUTHENTICATION")
    print("-" * 40)
    print("API Key required for prediction endpoints:")
    print("Header: Authorization: Bearer your_api_key_here")

    print("\n⚡ RATE LIMITS")
    print("-" * 40)
    print("Default: 100 requests per minute per API key")
    print("Headers: X-RateLimit-Limit, X-RateLimit-Remaining")

    print("\n💾 CACHING")
    print("-" * 40)
    print("Predictions cached for 5 minutes (Redis)")
    print("Cache status in response metadata")

    print("\n📈 MONITORING")
    print("-" * 40)
    print("Prometheus metrics at /metrics")
    print("Health checks at /health and /ready")
    print("Request tracing with X-Request-ID header")


if __name__ == "__main__":
    print_documentation()

    # Example usage
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)

    print("\n🐍 PYTHON EXAMPLE:")
    print(PYTHON_EXAMPLES["basic_usage"])

    print("\n🌐 CURL EXAMPLE:")
    print(CURL_EXAMPLES["single_prediction"])
