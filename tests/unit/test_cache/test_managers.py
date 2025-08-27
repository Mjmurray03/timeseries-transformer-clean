"""
Unit tests for Redis cache managers.

Tests basic CRUD operations, TTL handling, key generation consistency,
and error handling for all cache manager types.
"""

import asyncio
import json
import pickle
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import numpy as np

from src.cache.config import RedisConfig, create_test_config
from src.cache.connection import RedisConnectionManager
from src.cache.managers.prediction import PredictionCache
from src.cache.managers.feature import FeatureCache
from src.cache.managers.api import APICache
from src.cache.models import PredictionCacheEntry, FeatureCacheEntry, APICacheEntry
from src.cache.exceptions import CacheOperationError, CacheSerializationError


class TestPredictionCache:
    """Test suite for PredictionCache"""
    
    @pytest.fixture
    def config(self):
        """Create test Redis configuration"""
        return create_test_config()
    
    @pytest.fixture
    def connection_manager(self, config):
        """Create mock connection manager"""
        manager = Mock(spec=RedisConnectionManager)
        manager.config = config
        return manager
    
    @pytest.fixture
    def prediction_cache(self, connection_manager, config):
        """Create PredictionCache instance for testing"""
        return PredictionCache(connection_manager, config)
    
    def test_initialization(self, prediction_cache, config):
        """Test PredictionCache initializes correctly"""
        assert prediction_cache.cache_type == "prediction"
        assert prediction_cache.get_database() == config.prediction_cache_db
        assert prediction_cache.get_default_ttl() == config.prediction_cache_ttl
    
    def test_serialize_list_prediction(self, prediction_cache):
        """Test serialization of list predictions"""
        prediction = [100.5, 101.2, 99.8, 102.1, 103.0]
        
        serialized = prediction_cache.serialize(prediction)
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = prediction_cache.deserialize(serialized)
        assert deserialized == prediction
    
    def test_serialize_numpy_prediction(self, prediction_cache):
        """Test serialization of numpy array predictions"""
        prediction = np.array([100.5, 101.2, 99.8, 102.1, 103.0])
        
        serialized = prediction_cache.serialize(prediction)
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = prediction_cache.deserialize(serialized)
        np.testing.assert_array_equal(deserialized, prediction)
    
    def test_serialize_dict_prediction(self, prediction_cache):
        """Test serialization of dictionary predictions"""
        prediction = {
            "prices": [100.5, 101.2, 99.8],
            "confidence": [0.85, 0.82, 0.88],
            "volatility": 0.15
        }
        
        serialized = prediction_cache.serialize(prediction)
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = prediction_cache.deserialize(serialized)
        assert deserialized == prediction
    
    def test_serialize_compression(self, prediction_cache):
        """Test compression for large predictions"""
        # Create large prediction data (>1KB)
        large_prediction = [100.0 + i * 0.1 for i in range(10000)]
        
        serialized = prediction_cache.serialize(large_prediction)
        
        # Should be compressed (has compression marker)
        assert serialized.startswith(b'COMPRESSED:')
        
        # Test deserialization
        deserialized = prediction_cache.deserialize(serialized)
        assert deserialized == large_prediction
    
    def test_generate_prediction_key(self, prediction_cache):
        """Test prediction cache key generation"""
        key = prediction_cache.generate_prediction_key(
            ticker="AAPL",
            features_hash="abc123",
            model_version="v1.0.0",
            prediction_type="price"
        )
        
        assert key == "pred:AAPL:abc123:v1.0.0"
        
        # Test with different prediction type
        key2 = prediction_cache.generate_prediction_key(
            ticker="AAPL",
            features_hash="abc123",
            model_version="v1.0.0",
            prediction_type="direction"
        )
        
        assert key2 == "pred:AAPL:abc123:v1.0.0:direction"
    
    @pytest.mark.asyncio
    async def test_cache_prediction_success(self, prediction_cache):
        """Test successful prediction caching"""
        # Mock the set method
        prediction_cache.set = AsyncMock(return_value=True)
        
        prediction = [100.5, 101.2, 99.8, 102.1, 103.0]
        
        result = await prediction_cache.cache_prediction(
            ticker="AAPL",
            features_hash="abc123",
            prediction=prediction,
            model_version="v1.0.0",
            confidence_score=0.85
        )
        
        assert result is True
        prediction_cache.set.assert_called_once()
        
        # Check the cache entry that was passed to set
        call_args = prediction_cache.set.call_args
        cache_key, entry, ttl = call_args[0]
        
        assert cache_key == "pred:AAPL:abc123:v1.0.0"
        assert isinstance(entry, PredictionCacheEntry)
        assert entry.data == prediction
        assert entry.ticker == "AAPL"
        assert entry.model_version == "v1.0.0"
        assert entry.confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_get_prediction_success(self, prediction_cache):
        """Test successful prediction retrieval"""
        # Create mock prediction entry
        prediction = [100.5, 101.2, 99.8, 102.1, 103.0]
        entry = PredictionCacheEntry(
            data=prediction,
            created_at=datetime.now(),
            ttl=300,
            ticker="AAPL",
            model_version="v1.0.0",
            features_hash="abc123"
        )
        
        # Mock the get method
        prediction_cache.get = AsyncMock(return_value=entry)
        
        result = await prediction_cache.get_prediction(
            ticker="AAPL",
            features_hash="abc123",
            model_version="v1.0.0"
        )
        
        assert result == entry
        assert result.data == prediction
        prediction_cache.get.assert_called_once_with("pred:AAPL:abc123:v1.0.0")
    
    @pytest.mark.asyncio
    async def test_get_prediction_not_found(self, prediction_cache):
        """Test prediction retrieval when not found"""
        # Mock the get method to return None
        prediction_cache.get = AsyncMock(return_value=None)
        
        result = await prediction_cache.get_prediction(
            ticker="AAPL",
            features_hash="abc123"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_prediction_invalid_entry(self, prediction_cache):
        """Test prediction retrieval with invalid entry"""
        # Mock invalid entry (not PredictionCacheEntry)
        prediction_cache.get = AsyncMock(return_value="invalid_entry")
        
        result = await prediction_cache.get_prediction(
            ticker="AAPL",
            features_hash="abc123"
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_prediction_data_only(self, prediction_cache):
        """Test getting only prediction data"""
        prediction = [100.5, 101.2, 99.8, 102.1, 103.0]
        entry = PredictionCacheEntry(
            data=prediction,
            created_at=datetime.now(),
            ttl=300,
            ticker="AAPL",
            model_version="v1.0.0",
            features_hash="abc123"
        )
        
        # Mock get_prediction method
        prediction_cache.get_prediction = AsyncMock(return_value=entry)
        
        result = await prediction_cache.get_prediction_data(
            ticker="AAPL",
            features_hash="abc123"
        )
        
        assert result == prediction
    
    @pytest.mark.asyncio
    async def test_cache_batch_predictions(self, prediction_cache):
        """Test batch prediction caching"""
        predictions = [
            {
                "ticker": "AAPL",
                "features_hash": "abc123",
                "prediction": [100.5, 101.2],
                "model_version": "v1.0.0"
            },
            {
                "ticker": "GOOGL",
                "features_hash": "def456",
                "prediction": [2800.0, 2850.0],
                "model_version": "v1.0.0"
            }
        ]
        
        # Mock cache_prediction method
        prediction_cache.cache_prediction = AsyncMock(return_value=True)
        
        results = await prediction_cache.cache_batch_predictions(predictions)
        
        assert len(results) == 2
        assert all(results.values())  # All should be True
        assert prediction_cache.cache_prediction.call_count == 2


class TestFeatureCache:
    """Test suite for FeatureCache"""
    
    @pytest.fixture
    def config(self):
        """Create test Redis configuration"""
        return create_test_config()
    
    @pytest.fixture
    def connection_manager(self, config):
        """Create mock connection manager"""
        manager = Mock(spec=RedisConnectionManager)
        manager.config = config
        return manager
    
    @pytest.fixture
    def feature_cache(self, connection_manager, config):
        """Create FeatureCache instance for testing"""
        return FeatureCache(connection_manager, config)
    
    def test_initialization(self, feature_cache, config):
        """Test FeatureCache initializes correctly"""
        assert feature_cache.cache_type == "feature"
        assert feature_cache.get_database() == config.feature_cache_db
        assert feature_cache.get_default_ttl() == config.feature_cache_ttl
    
    def test_serialize_dataframe(self, feature_cache):
        """Test DataFrame serialization"""
        df = pd.DataFrame({
            'rsi': [65.5, 70.2, 68.1],
            'macd': [1.23, 1.45, 1.12],
            'volume': [1000000, 1200000, 950000]
        })
        
        serialized = feature_cache.serialize(df)
        assert isinstance(serialized, bytes)
        # May be compressed if large, so check for either marker
        assert (serialized.startswith(b'DATAFRAME:') or 
                serialized.startswith(b'COMPRESSED:'))
        
        # Test deserialization
        deserialized = feature_cache.deserialize(serialized)
        pd.testing.assert_frame_equal(deserialized, df)
    
    def test_serialize_numpy_array(self, feature_cache):
        """Test numpy array serialization"""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        serialized = feature_cache.serialize(arr)
        assert isinstance(serialized, bytes)
        assert serialized.startswith(b'NUMPY:')
        
        # Test deserialization
        deserialized = feature_cache.deserialize(serialized)
        np.testing.assert_array_equal(deserialized, arr)
    
    def test_serialize_dict_features(self, feature_cache):
        """Test dictionary features serialization"""
        features = {
            'rsi': 65.5,
            'macd': 1.23,
            'bb_upper': 105.2,
            'bb_lower': 98.7
        }
        
        serialized = feature_cache.serialize(features)
        assert isinstance(serialized, bytes)
        assert serialized.startswith(b'PICKLE:')
        
        # Test deserialization
        deserialized = feature_cache.deserialize(serialized)
        assert deserialized == features
    
    def test_generate_feature_key(self, feature_cache):
        """Test feature cache key generation"""
        key = feature_cache.generate_feature_key(
            ticker="AAPL",
            date_range=("2024-01-01", "2024-01-31"),
            feature_types=["rsi", "macd", "bollinger"]
        )
        
        # Should include ticker, date range, and feature types hash
        assert key.startswith("feat:AAPL:2024-01-01_2024-01-31:")
        assert len(key.split(":")) == 4  # feat:ticker:dates:hash
    
    def test_generate_feature_key_datetime(self, feature_cache):
        """Test feature key generation with datetime objects"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        key = feature_cache.generate_feature_key(
            ticker="AAPL",
            date_range=(start_date, end_date)
        )
        
        assert "feat:AAPL:2024-01-01_2024-01-31" in key
    
    @pytest.mark.asyncio
    async def test_cache_features_success(self, feature_cache):
        """Test successful feature caching"""
        # Mock the set method
        feature_cache.set = AsyncMock(return_value=True)
        
        df = pd.DataFrame({
            'rsi': [65.5, 70.2],
            'macd': [1.23, 1.45]
        })
        
        result = await feature_cache.cache_features(
            ticker="AAPL",
            date_range=("2024-01-01", "2024-01-31"),
            features=df,
            feature_types=["rsi", "macd"]
        )
        
        assert result is True
        feature_cache.set.assert_called_once()
        
        # Check the cache entry
        call_args = feature_cache.set.call_args
        cache_key, entry, ttl = call_args[0]
        
        assert isinstance(entry, FeatureCacheEntry)
        pd.testing.assert_frame_equal(entry.data, df)
        assert entry.ticker == "AAPL"
        assert entry.feature_types == ["rsi", "macd"]
    
    @pytest.mark.asyncio
    async def test_get_features_success(self, feature_cache):
        """Test successful feature retrieval"""
        df = pd.DataFrame({'rsi': [65.5, 70.2]})
        entry = FeatureCacheEntry(
            data=df,
            created_at=datetime.now(),
            ttl=3600,
            ticker="AAPL",
            date_range_start=datetime(2024, 1, 1),
            date_range_end=datetime(2024, 1, 31),
            feature_types=["rsi"]
        )
        
        # Mock the get method
        feature_cache.get = AsyncMock(return_value=entry)
        
        result = await feature_cache.get_features(
            ticker="AAPL",
            date_range=("2024-01-01", "2024-01-31")
        )
        
        assert result == entry
        pd.testing.assert_frame_equal(result.data, df)


class TestAPICache:
    """Test suite for APICache"""
    
    @pytest.fixture
    def config(self):
        """Create test Redis configuration"""
        return create_test_config()
    
    @pytest.fixture
    def connection_manager(self, config):
        """Create mock connection manager"""
        manager = Mock(spec=RedisConnectionManager)
        manager.config = config
        return manager
    
    @pytest.fixture
    def api_cache(self, connection_manager, config):
        """Create APICache instance for testing"""
        return APICache(connection_manager, config)
    
    def test_initialization(self, api_cache, config):
        """Test APICache initializes correctly"""
        assert api_cache.cache_type == "api"
        assert api_cache.get_database() == config.api_cache_db
        assert api_cache.get_default_ttl() == config.api_cache_ttl
    
    def test_serialize_dict_response(self, api_cache):
        """Test dictionary response serialization"""
        response = {
            "status": "success",
            "data": {"prediction": [100.5, 101.2]},
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        serialized = api_cache.serialize(response)
        assert isinstance(serialized, str)
        
        # Test deserialization
        deserialized = api_cache.deserialize(serialized)
        assert deserialized == response
    
    def test_serialize_string_response(self, api_cache):
        """Test string response serialization"""
        response = "Simple string response"
        
        serialized = api_cache.serialize(response)
        assert serialized == response
        
        # Test deserialization
        deserialized = api_cache.deserialize(serialized)
        assert deserialized == response
    
    def test_generate_request_hash(self, api_cache):
        """Test request hash generation"""
        hash1 = api_cache.generate_request_hash(
            method="GET",
            endpoint="/api/predict",
            query_params={"ticker": "AAPL", "days": "5"}
        )
        
        hash2 = api_cache.generate_request_hash(
            method="GET",
            endpoint="/api/predict",
            query_params={"ticker": "AAPL", "days": "5"}
        )
        
        # Same parameters should generate same hash
        assert hash1 == hash2
        
        # Different parameters should generate different hash
        hash3 = api_cache.generate_request_hash(
            method="GET",
            endpoint="/api/predict",
            query_params={"ticker": "GOOGL", "days": "5"}
        )
        
        assert hash1 != hash3
    
    def test_generate_api_cache_key(self, api_cache):
        """Test API cache key generation"""
        key = api_cache.generate_api_cache_key(
            method="GET",
            endpoint="/api/predict",
            request_hash="abc123"
        )
        
        assert key == "api:get:api_predict:abc123"
    
    @pytest.mark.asyncio
    async def test_cache_response_success(self, api_cache):
        """Test successful API response caching"""
        # Mock the set method
        api_cache.set = AsyncMock(return_value=True)
        
        response_data = {"prediction": [100.5, 101.2]}
        
        result = await api_cache.cache_response(
            method="GET",
            endpoint="/api/predict",
            response_data=response_data,
            status_code=200,
            query_params={"ticker": "AAPL"}
        )
        
        assert result is True
        api_cache.set.assert_called_once()
        
        # Check the cache entry
        call_args = api_cache.set.call_args
        cache_key, entry, ttl = call_args[0]
        
        assert isinstance(entry, APICacheEntry)
        assert entry.data == response_data
        assert entry.endpoint == "/api/predict"
        assert entry.method == "GET"
        assert entry.status_code == 200
    
    @pytest.mark.asyncio
    async def test_get_cached_response_success(self, api_cache):
        """Test successful cached response retrieval"""
        response_data = {"prediction": [100.5, 101.2]}
        entry = APICacheEntry(
            data=response_data,
            created_at=datetime.now(),
            ttl=300,
            endpoint="/api/predict",
            method="GET",
            status_code=200,
            request_hash="abc123"
        )
        
        # Mock the get method
        api_cache.get = AsyncMock(return_value=entry)
        
        result = await api_cache.get_cached_response(
            method="GET",
            endpoint="/api/predict",
            query_params={"ticker": "AAPL"}
        )
        
        assert result == entry
        assert result.data == response_data
    
    @pytest.mark.asyncio
    async def test_get_response_data_only(self, api_cache):
        """Test getting only response data"""
        response_data = {"prediction": [100.5, 101.2]}
        entry = APICacheEntry(
            data=response_data,
            created_at=datetime.now(),
            ttl=300,
            endpoint="/api/predict",
            method="GET",
            status_code=200,
            request_hash="abc123"
        )
        
        # Mock get_cached_response method
        api_cache.get_cached_response = AsyncMock(return_value=entry)
        
        result = await api_cache.get_response_data(
            method="GET",
            endpoint="/api/predict",
            query_params={"ticker": "AAPL"}
        )
        
        assert result == response_data


class TestCacheManagerErrorHandling:
    """Test error handling across all cache managers"""
    
    @pytest.fixture
    def config(self):
        return create_test_config()
    
    @pytest.fixture
    def connection_manager(self, config):
        manager = Mock(spec=RedisConnectionManager)
        manager.config = config
        return manager
    
    @pytest.mark.asyncio
    async def test_prediction_cache_serialization_error(self, connection_manager, config):
        """Test prediction cache serialization error handling"""
        cache = PredictionCache(connection_manager, config)
        
        # Create object that can't be pickled
        class UnpicklableObject:
            def __reduce__(self):
                raise TypeError("Cannot pickle this object")
        
        unpicklable = UnpicklableObject()
        
        with pytest.raises(CacheSerializationError):
            cache.serialize(unpicklable)
    
    @pytest.mark.asyncio
    async def test_feature_cache_deserialization_error(self, connection_manager, config):
        """Test feature cache deserialization error handling"""
        cache = FeatureCache(connection_manager, config)
        
        # Invalid serialized data
        invalid_data = b"invalid_serialized_data"
        
        with pytest.raises(CacheSerializationError):
            cache.deserialize(invalid_data)
    
    @pytest.mark.asyncio
    async def test_api_cache_json_error(self, connection_manager, config):
        """Test API cache JSON serialization error"""
        cache = APICache(connection_manager, config)
        
        # Create object that can't be JSON serialized
        class NonJSONSerializable:
            pass
        
        # This should fall back to string representation
        result = cache.serialize(NonJSONSerializable())
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__])