"""
Time-Series Transformer API

Production-ready FastAPI application for stock price prediction using transformer models.
Provides endpoints for single and batch predictions with comprehensive error handling,
authentication, rate limiting, caching, and monitoring.
"""

from .main import app
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
    ErrorResponse
)
from .model_server import ModelServer, ModelPool
from .cache import PredictionCache, ModelCache, initialize_caches
from .exceptions import (
    BaseAPIException,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    InferenceError,
    ServiceUnavailableError
)

__version__ = "1.0.0"
__all__ = [
    "app",
    "PredictionRequest",
    "PredictionResponse", 
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "MetricsResponse",
    "ErrorResponse",
    "ModelServer",
    "ModelPool",
    "PredictionCache",
    "ModelCache",
    "initialize_caches",
    "BaseAPIException",
    "ValidationError",
    "AuthenticationError", 
    "RateLimitError",
    "InferenceError",
    "ServiceUnavailableError"
]