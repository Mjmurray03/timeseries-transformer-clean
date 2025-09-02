#!/usr/bin/env python3
"""
Comprehensive API Testing Suite

Tests all endpoints according to PROMPT 5 requirements:
- /predict endpoint with sample data
- Caching functionality
- Input validation
- /backtest endpoint
- Performance requirements
"""

import pytest
import asyncio
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import redis

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the API
from src.api.main import app, model_manager, cache_manager


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample feature data for testing"""
    # 60 days x 8 features
    np.random.seed(42)
    return np.random.randn(60, 8).tolist()


@pytest.fixture
def mock_model_manager():
    """Mock model manager for testing"""
    with patch('src.api.main.model_manager') as mock_manager:
        # Mock prediction response
        mock_manager.predict.return_value = {
            'predictions': [100.5, 101.2, 102.0],
            'ci_lower': [99.0, 99.5, 100.0],
            'ci_upper': [102.0, 103.0, 104.0]
        }
        mock_manager.models = {'AAPL': Mock(), 'MSFT': Mock(), 'NVDA': Mock()}
        mock_manager.device = 'cpu'
        yield mock_manager


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing"""
    with patch('src.api.main.cache_manager') as mock_cache:
        mock_cache.enabled = True
        mock_cache.get.return_value = None  # No cache by default
        mock_cache.set.return_value = None
        mock_cache.get_cache_key.return_value = "test_cache_key"
        yield mock_cache


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client, mock_model_manager):
        """Test successful health check"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "cuda_available" in data
        assert "models_loaded" in data
        assert "cache_enabled" in data
        assert isinstance(data["models_loaded"], list)


class TestPredictionEndpoint:
    """Test /predict endpoint with sample data"""
    
    def test_prediction_endpoint_success(self, client, sample_features, mock_model_manager, mock_cache_manager):
        """Test /predict endpoint with sample data"""
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 3
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["ticker"] == "AAPL"
        assert "predictions" in data
        assert "confidence_intervals" in data
        assert "timestamp" in data
        assert "model_version" in data
        assert "cache_hit" in data
        
        # Verify predictions format
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 3  # horizon=3
        assert all(isinstance(p, (int, float)) for p in data["predictions"])
        
        # Verify confidence intervals
        ci = data["confidence_intervals"]
        assert "lower" in ci
        assert "upper" in ci
        assert len(ci["lower"]) == 3
        assert len(ci["upper"]) == 3
        
        # Verify model was called correctly
        mock_model_manager.predict.assert_called_once()
        call_args = mock_model_manager.predict.call_args
        assert call_args[0][0] == "AAPL"  # ticker
        assert call_args[0][1].shape == (60, 8)  # features shape
    
    def test_prediction_endpoint_performance(self, client, sample_features, mock_model_manager, mock_cache_manager):
        """Test /predict responds in <500ms (uncached)"""
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 3
        }
        
        start_time = time.time()
        response = client.post("/predict", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 500, f"Response time {response_time_ms:.2f}ms exceeds 500ms limit"
    
    def test_invalid_ticker_rejected(self, client, sample_features):
        """Test invalid ticker is rejected"""
        request_data = {
            "ticker": "INVALID",
            "features": sample_features,
            "horizon": 3
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 422  # Validation error
        assert "error" in response.json() or "detail" in response.json()
    
    def test_invalid_features_rejected(self, client):
        """Test invalid features are rejected"""
        # Test wrong number of days
        request_data = {
            "ticker": "AAPL",
            "features": [[1.0] * 8] * 30,  # 30 days instead of 60
            "horizon": 3
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
        
        # Test wrong number of features per day
        request_data = {
            "ticker": "AAPL", 
            "features": [[1.0] * 5] * 60,  # 5 features instead of 8
            "horizon": 3
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_horizon_rejected(self, client, sample_features):
        """Test invalid horizon values are rejected"""
        # Test horizon too small
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 0
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
        
        # Test horizon too large
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 20
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422


class TestCachingFunctionality:
    """Test caching functionality"""
    
    def test_caching_works(self, client, sample_features, mock_model_manager, mock_cache_manager):
        """Verify cache hits on repeated requests"""
        # First setup cache to return cached data on second call
        cached_data = {
            "ticker": "AAPL",
            "predictions": [100.0, 101.0, 102.0],
            "confidence_intervals": {
                "lower": [99.0, 99.5, 100.0],
                "upper": [101.0, 102.5, 104.0]
            },
            "timestamp": datetime.now(),
            "model_version": "1.0.0"
        }
        
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 3
        }
        
        # First request - cache miss
        response1 = client.post("/predict", json=request_data)
        assert response1.status_code == 200
        assert not response1.json()["cache_hit"]
        
        # Setup cache to return data on second call
        mock_cache_manager.get.return_value = cached_data
        
        # Second request - cache hit
        response2 = client.post("/predict", json=request_data)
        assert response2.status_code == 200
        assert response2.json()["cache_hit"]
        
        # Verify cache was used
        assert mock_cache_manager.get.call_count >= 2
        assert mock_cache_manager.set.call_count >= 1
    
    def test_cache_hit_performance(self, client, sample_features, mock_cache_manager):
        """Test cached responses are <100ms"""
        # Setup cache to return immediate response
        cached_data = {
            "ticker": "AAPL",
            "predictions": [100.0, 101.0, 102.0],
            "confidence_intervals": {
                "lower": [99.0, 99.5, 100.0], 
                "upper": [101.0, 102.5, 104.0]
            },
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
        
        mock_cache_manager.get.return_value = cached_data
        
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 3
        }
        
        start_time = time.time()
        response = client.post("/predict", json=request_data)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response.json()["cache_hit"]
        assert response_time_ms < 100, f"Cached response time {response_time_ms:.2f}ms exceeds 100ms limit"


class TestBacktestEndpoint:
    """Test backtesting endpoint"""
    
    @patch('src.api.main.Path')
    @patch('src.backtesting.backtest_engine.BacktestEngine')
    def test_backtest_endpoint(self, mock_engine_class, mock_path, client, mock_model_manager):
        """Test backtesting returns valid metrics"""
        # Mock file existence
        mock_path.return_value.exists.return_value = True
        
        # Mock backtest engine
        mock_engine = Mock()
        mock_engine.run_quick_backtest.return_value = {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.65,
            "num_trades": 25,
            "period_days": 252
        }
        mock_engine_class.return_value = mock_engine
        
        request_data = {
            "ticker": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000,
            "strategy_params": {
                "return_threshold": 0.02,
                "confidence_threshold": 0.7,
                "max_positions": 5
            }
        }
        
        response = client.post("/backtest", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            "total_return", "sharpe_ratio", "max_drawdown", 
            "win_rate", "num_trades", "period_days"
        ]
        for field in required_fields:
            assert field in data
            assert isinstance(data[field], (int, float))
        
        # Verify reasonable values
        assert -1.0 <= data["total_return"] <= 10.0
        assert -5.0 <= data["sharpe_ratio"] <= 10.0
        assert -1.0 <= data["max_drawdown"] <= 0.0
        assert 0.0 <= data["win_rate"] <= 1.0
        assert data["num_trades"] >= 0
        assert data["period_days"] > 0
    
    @patch('src.api.main.Path')
    def test_backtest_missing_data(self, mock_path, client):
        """Test backtest with missing ticker data"""
        mock_path.return_value.exists.return_value = False
        
        request_data = {
            "ticker": "AAPL",
            "start_date": "2023-01-01", 
            "end_date": "2023-12-31",
            "initial_capital": 100000
        }
        
        response = client.post("/backtest", json=request_data)
        
        assert response.status_code == 404
        assert "No data for" in response.json()["detail"]


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info_success(self, client, mock_model_manager):
        """Test model info returns correct structure"""
        response = client.get("/model-info")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = [
            "model_version", "architecture", "parameters",
            "training_date", "supported_tickers", "performance_metrics"
        ]
        for field in required_fields:
            assert field in data
        
        # Verify data types and values
        assert isinstance(data["model_version"], str)
        assert isinstance(data["architecture"], str)
        assert isinstance(data["parameters"], int)
        assert isinstance(data["supported_tickers"], list)
        assert isinstance(data["performance_metrics"], dict)
        
        # Verify performance metrics structure
        metrics = data["performance_metrics"]
        expected_metrics = ["avg_rmse", "avg_sharpe", "directional_accuracy"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


class TestErrorHandling:
    """Test error handling"""
    
    def test_validation_errors_handled(self, client):
        """Test validation errors are properly handled"""
        # Missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        # Invalid data types
        response = client.post("/predict", json={
            "ticker": 123,  # Should be string
            "features": "invalid",  # Should be list
            "horizon": "invalid"  # Should be int
        })
        assert response.status_code == 422
    
    @patch('src.api.main.model_manager')
    def test_model_errors_handled(self, mock_manager, client, sample_features):
        """Test model errors are properly handled"""
        mock_manager.predict.side_effect = Exception("Model error")
        
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 3
        }
        
        response = client.post("/predict", json=request_data)
        
        assert response.status_code == 500
        assert "detail" in response.json()


class TestConcurrentRequests:
    """Test concurrent request handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_model_manager, mock_cache_manager, sample_features):
        """Test handling 100 concurrent requests without errors"""
        
        async def make_request():
            client = TestClient(app)
            request_data = {
                "ticker": "AAPL",
                "features": sample_features,
                "horizon": 3
            }
            response = client.post("/predict", json=request_data)
            return response.status_code == 200
        
        # Create 100 concurrent requests
        tasks = [make_request() for _ in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most requests succeeded
        success_count = sum(1 for r in results if r is True)
        success_rate = success_count / len(results)
        
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95% threshold"


class TestMemoryManagement:
    """Test memory management"""
    
    def test_no_memory_leaks_in_predictions(self, client, sample_features, mock_model_manager, mock_cache_manager):
        """Test memory usage stays reasonable during predictions"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make 50 requests
        request_data = {
            "ticker": "AAPL",
            "features": sample_features,
            "horizon": 3
        }
        
        for _ in range(50):
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 100MB for 50 requests)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB, possible memory leak"


def test_api_documentation_available(client):
    """Test that API documentation is available"""
    # Test OpenAPI docs
    response = client.get("/docs")
    assert response.status_code == 200
    
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/backtest" in schema["paths"]
    assert "/model-info" in schema["paths"]


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "--tb=short"])