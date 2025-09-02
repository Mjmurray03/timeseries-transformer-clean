import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch

from .cache import get_model_cache
from .schemas import ConfidenceInterval, PredictionMetadata, PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class ModelServer:
    """Manages model loading, inference, and preprocessing"""

    def __init__(self, model_path: str, scaler_path: Optional[str] = None, device: str = "auto"):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.device = self._setup_device(device)
        self.model = None
        self.scaler = None
        self.model_info = {}
        self.loaded_at = None

        logger.info(f"Initializing ModelServer with device: {self.device}")
        self._load_components()
        self._warm_up()

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("CUDA not available. Using CPU.")

        return torch.device(device)

    def _load_components(self):
        """Load model and preprocessing components"""
        try:
            # Check model cache first
            cache = get_model_cache()
            cache_key = f"model:{self.model_path.name}"

            if cache:
                cached_model = cache.get(cache_key)
                if cached_model:
                    self.model = cached_model
                    logger.info(f"Loaded model from cache: {self.model_path.name}")
                else:
                    self.model = self._load_model()
                    cache.set(cache_key, self.model)
            else:
                self.model = self._load_model()

            # Load scaler if provided
            if self.scaler_path and self.scaler_path.exists():
                self.scaler = self._load_scaler()
            else:
                logger.warning("No scaler provided. Input normalization will be skipped.")

            self.loaded_at = datetime.now()
            logger.info("Model components loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model components: {e}")
            raise

    def _load_model(self) -> torch.nn.Module:
        """Load PyTorch model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            # Load TorchScript model for production
            if self.model_path.suffix == ".pt":
                model = torch.jit.load(str(self.model_path), map_location=self.device)
                logger.info(f"Loaded TorchScript model from {self.model_path}")
            else:
                # Load regular PyTorch model
                model = torch.load(str(self.model_path), map_location=self.device)
                logger.info(f"Loaded PyTorch model from {self.model_path}")

            model.eval()

            # Extract model info
            self.model_info = {
                "path": str(self.model_path),
                "device": str(self.device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "model_size_mb": self.model_path.stat().st_size / (1024 * 1024),
            }

            return model

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def _load_scaler(self):
        """Load feature scaler"""
        try:
            scaler = joblib.load(self.scaler_path)
            logger.info(f"Loaded scaler from {self.scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"Failed to load scaler from {self.scaler_path}: {e}")
            raise

    def _warm_up(self):
        """Warm up model with dummy predictions"""
        if not self.model:
            return

        logger.info("Warming up model...")
        dummy_input = torch.randn(1, 60, 7, device=self.device)

        # Run several warm-up iterations
        warmup_times = []
        for i in range(10):
            start = time.time()
            with torch.no_grad():
                _ = self.model(dummy_input)
            warmup_times.append(time.time() - start)

        avg_warmup_time = np.mean(warmup_times) * 1000
        logger.info(f"Model warmed up. Average inference time: {avg_warmup_time:.2f}ms")

    def preprocess_features(self, features: list) -> torch.Tensor:
        """Preprocess input features"""
        try:
            # Convert to numpy array
            features_np = np.array(features, dtype=np.float32)

            # Validate shape
            if features_np.shape != (60, 7):
                raise ValueError(f"Invalid input shape: {features_np.shape}, expected (60, 7)")

            # Check for invalid values
            if np.any(np.isnan(features_np)) or np.any(np.isinf(features_np)):
                raise ValueError("Features contain NaN or infinite values")

            # Apply scaling if scaler is available
            if self.scaler:
                # Reshape for scaling (flatten time dimension)
                features_flat = features_np.reshape(-1, 7)
                features_scaled = self.scaler.transform(features_flat)
                features_np = features_scaled.reshape(60, 7)

            # Convert to tensor and add batch dimension
            features_tensor = torch.from_numpy(features_np).unsqueeze(0).to(self.device)

            return features_tensor

        except Exception as e:
            logger.error(f"Feature preprocessing failed: {e}")
            raise

    def postprocess_output(
        self, output: torch.Tensor, ticker: str, horizon: int
    ) -> Tuple[list, Dict[str, Any]]:
        """Postprocess model output"""
        try:
            # Move to CPU and convert to numpy
            if isinstance(output, dict):
                # Model returns dictionary with predictions and attention
                predictions = output["predictions"].cpu().numpy()
                attention_weights = output.get("attention_weights")
                if attention_weights is not None:
                    attention_weights = attention_weights.cpu().numpy()
            else:
                # Model returns tensor directly
                predictions = output.cpu().numpy()
                attention_weights = None

            # Extract predictions for requested horizon
            if predictions.shape[0] == 1:  # Remove batch dimension
                predictions = predictions[0]

            # Take only the requested horizon length
            if len(predictions) >= horizon:
                predictions = predictions[:horizon]
            else:
                # Pad if necessary (shouldn't happen with proper model)
                predictions = np.pad(
                    predictions,
                    (0, horizon - len(predictions)),
                    mode="constant",
                    constant_values=predictions[-1],
                )

            # Convert to list
            prediction_list = predictions.tolist()

            # Process attention weights if available
            attention_list = None
            if attention_weights is not None:
                if attention_weights.ndim == 3:  # Remove batch dimension
                    attention_weights = attention_weights[0]
                attention_list = attention_weights.tolist()

            # Generate confidence intervals (placeholder implementation)
            confidence_intervals = self._generate_confidence_intervals(prediction_list, horizon)

            return prediction_list, {
                "attention_weights": attention_list,
                "confidence_intervals": confidence_intervals,
            }

        except Exception as e:
            logger.error(f"Output postprocessing failed: {e}")
            raise

    def _generate_confidence_intervals(
        self, predictions: list, horizon: int
    ) -> Dict[str, ConfidenceInterval]:
        """Generate confidence intervals for predictions"""
        # Placeholder implementation - in production, this would use model uncertainty
        predictions_array = np.array(predictions)

        # Simple confidence intervals based on prediction magnitude
        # In practice, you'd use model-specific uncertainty estimation
        std_dev = np.std(predictions_array) if len(predictions_array) > 1 else 0.1

        confidence_intervals = {}

        for confidence_level in [0.68, 0.95]:  # 68% and 95% confidence intervals
            z_score = 1.0 if confidence_level == 0.68 else 1.96
            margin = z_score * std_dev

            lower = (predictions_array - margin).tolist()
            upper = (predictions_array + margin).tolist()

            confidence_intervals[f"{int(confidence_level * 100)}%"] = ConfidenceInterval(
                lower=lower, upper=upper, confidence_level=confidence_level
            )

        return confidence_intervals

    @torch.no_grad()
    def predict(self, request: PredictionRequest, request_id: str) -> PredictionResponse:
        """Run inference and return prediction response"""
        if not self.model:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Preprocess features
            features_tensor = self.preprocess_features(request.features)

            # Run inference
            inference_start = time.time()

            if hasattr(self.model, "forward_with_attention"):
                # Model supports attention output
                output = self.model.forward_with_attention(features_tensor)
            else:
                # Standard forward pass
                output = self.model(features_tensor)

            inference_time = (time.time() - inference_start) * 1000

            # Postprocess output
            predictions, extra_data = self.postprocess_output(
                output, request.ticker, request.horizon
            )

            # Create metadata
            total_time = (time.time() - start_time) * 1000
            metadata = PredictionMetadata(
                model_version=self.get_model_version(),
                inference_time_ms=inference_time,
                timestamp=datetime.now(),
                cache_hit=False,
                request_id=request_id,
            )

            # Create response
            response = PredictionResponse(
                prediction=predictions,
                confidence_intervals=extra_data.get("confidence_intervals", {}),
                attention_weights=extra_data.get("attention_weights"),
                metadata=metadata,
            )

            logger.debug(f"Prediction completed in {total_time:.2f}ms for {request.ticker}")
            return response

        except Exception as e:
            logger.error(f"Prediction failed for {request.ticker}: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        return {
            **self.model_info,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "scaler_available": self.scaler is not None,
            "device": str(self.device),
        }

    def get_model_version(self) -> str:
        """Extract model version"""
        # Try to get version from model if available
        if hasattr(self.model, "version"):
            return str(self.model.version)

        # Fall back to filename-based versioning
        stem = self.model_path.stem
        if "v" in stem.lower():
            version_part = stem.lower().split("v")[-1]
            return f"v{version_part}"

        return "unknown"

    def health_check(self) -> bool:
        """Check if model server is healthy"""
        try:
            if not self.model:
                return False

            # Quick inference test
            dummy_input = torch.randn(1, 60, 7, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)

            return True

        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return False


class ModelPool:
    """Manages multiple model instances for load balancing"""

    def __init__(self, model_configs: list, pool_size: int = 1):
        self.instances = []
        self.current_index = 0

        for i in range(pool_size):
            for config in model_configs:
                instance = ModelServer(**config)
                self.instances.append(instance)

        logger.info(f"Initialized model pool with {len(self.instances)} instances")

    def get_instance(self) -> ModelServer:
        """Get next available model instance (round-robin)"""
        instance = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return instance

    def health_check(self) -> Dict[str, bool]:
        """Check health of all instances"""
        health_status = {}
        for i, instance in enumerate(self.instances):
            health_status[f"instance_{i}"] = instance.health_check()
        return health_status

    def get_pool_info(self) -> Dict[str, Any]:
        """Get information about the model pool"""
        return {
            "pool_size": len(self.instances),
            "current_index": self.current_index,
            "instances": [instance.get_model_info() for instance in self.instances],
        }
