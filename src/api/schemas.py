import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Request schema for single predictions"""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    features: List[List[float]] = Field(..., description="60 time steps with 7 features each")
    horizon: Optional[int] = Field(default=5, description="Prediction horizon (1-30 days)")

    @validator("ticker")
    def validate_ticker(cls, v):
        if not isinstance(v, str):
            raise ValueError("Ticker must be a string")
        if not v.isupper():
            raise ValueError("Ticker must be uppercase")
        if not re.match(r"^[A-Z]{1,5}$", v):
            raise ValueError("Ticker must be 1-5 uppercase letters")
        return v

    @validator("features")
    def validate_features(cls, v):
        if not isinstance(v, list):
            raise ValueError("Features must be a list")
        if len(v) != 60:
            raise ValueError("Features must have exactly 60 time steps")

        for i, row in enumerate(v):
            if not isinstance(row, list):
                raise ValueError(f"Time step {i} must be a list")
            if len(row) != 7:
                raise ValueError(f"Time step {i} must have exactly 7 features, got {len(row)}")

            for j, feature in enumerate(row):
                if not isinstance(feature, (int, float)):
                    raise ValueError(f"Feature at time step {i}, position {j} must be numeric")
                if abs(feature) > 10:
                    raise ValueError(
                        f"Feature at time step {i}, position {j} is out of range [-10, 10]"
                    )
                if feature != feature:  # Check for NaN
                    raise ValueError(f"Feature at time step {i}, position {j} cannot be NaN")

        return v

    @validator("horizon")
    def validate_horizon(cls, v):
        if v is not None:
            if not isinstance(v, int):
                raise ValueError("Horizon must be an integer")
            if v < 1 or v > 30:
                raise ValueError("Horizon must be between 1 and 30 days")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""

    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")

    @validator("requests")
    def validate_batch_size(cls, v):
        if not isinstance(v, list):
            raise ValueError("Requests must be a list")
        if len(v) == 0:
            raise ValueError("At least one request is required")
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100 requests")
        return v


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions"""

    lower: List[float] = Field(..., description="Lower bound of confidence interval")
    upper: List[float] = Field(..., description="Upper bound of confidence interval")
    confidence_level: float = Field(0.95, description="Confidence level (e.g., 0.95 for 95%)")


class PredictionMetadata(BaseModel):
    """Metadata for prediction response"""

    model_version: str = Field(..., description="Version of the model used")
    inference_time_ms: float = Field(..., description="Time taken for inference in milliseconds")
    timestamp: datetime = Field(..., description="Timestamp when prediction was made")
    cache_hit: bool = Field(False, description="Whether result was served from cache")
    request_id: str = Field(..., description="Unique identifier for this request")


class PredictionResponse(BaseModel):
    """Response schema for single predictions"""

    prediction: List[float] = Field(..., description="Predicted values for the horizon")
    confidence_intervals: Dict[str, ConfidenceInterval] = Field(
        default_factory=dict, description="Confidence intervals at different levels"
    )
    attention_weights: Optional[List[List[float]]] = Field(
        default=None, description="Attention weights from transformer model"
    )
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""

    predictions: List[PredictionResponse] = Field(..., description="List of prediction responses")
    batch_metadata: Dict[str, Any] = Field(..., description="Batch-level metadata")


class HealthResponse(BaseModel):
    """Response schema for health checks"""

    status: str = Field(..., description="Service status (healthy/unhealthy)")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    model_status: Dict[str, str] = Field(..., description="Status of loaded models")
    dependencies: Dict[str, str] = Field(..., description="Status of external dependencies")


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""

    model_version: str = Field(..., description="Model version")
    architecture: str = Field(..., description="Model architecture type")
    parameters: int = Field(..., description="Number of model parameters")
    device: str = Field(..., description="Device model is running on")
    loaded_at: datetime = Field(..., description="When model was loaded")
    training_metrics: Dict[str, float] = Field(..., description="Model training metrics")


class MetricsResponse(BaseModel):
    """Response schema for metrics endpoint"""

    active_connections: int = Field(..., description="Number of active connections")
    total_requests: int = Field(..., description="Total requests served")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    avg_inference_time_ms: float = Field(..., description="Average inference time in ms")
    error_rate: float = Field(..., description="Error rate percentage")


class ErrorDetail(BaseModel):
    """Detailed error information"""

    error_code: str = Field(..., description="Specific error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class ErrorResponse(BaseModel):
    """Standardized error response"""

    error: ErrorDetail = Field(..., description="Error information")
    timestamp: datetime = Field(..., description="Error timestamp")


class WebSocketMessage(BaseModel):
    """WebSocket message for streaming predictions"""

    type: str = Field(..., description="Message type (prediction, error, status)")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Message payload")
    timestamp: datetime = Field(..., description="Message timestamp")
    request_id: str = Field(..., description="Request identifier")


class StreamingRequest(BaseModel):
    """Request for streaming predictions"""

    ticker: str = Field(..., description="Stock ticker to stream")
    update_interval: int = Field(default=5, description="Update interval in seconds")
    features_source: str = Field(default="live", description="Source of feature data")

    @validator("update_interval")
    def validate_update_interval(cls, v):
        if v < 1 or v > 300:
            raise ValueError("Update interval must be between 1 and 300 seconds")
        return v


class AuthRequest(BaseModel):
    """Authentication request"""

    api_key: str = Field(..., description="API key for authentication")

    @validator("api_key")
    def validate_api_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError("Invalid API key format")
        return v


class AuthResponse(BaseModel):
    """Authentication response"""

    authenticated: bool = Field(..., description="Whether authentication was successful")
    user_id: Optional[str] = Field(default=None, description="Authenticated user ID")
    rate_limit: Optional[int] = Field(default=None, description="Rate limit for this user")
    expires_at: Optional[datetime] = Field(default=None, description="When authentication expires")


class RateLimitInfo(BaseModel):
    """Rate limit information"""

    limit: int = Field(..., description="Request limit per window")
    remaining: int = Field(..., description="Remaining requests in current window")
    reset_at: datetime = Field(..., description="When the rate limit window resets")
    retry_after: Optional[int] = Field(default=None, description="Seconds to wait before retry")
