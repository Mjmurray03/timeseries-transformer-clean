from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import asyncio
import logging
import time
import os
import redis
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import json

from .schemas import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse, MetricsResponse, WebSocketMessage, StreamingRequest,
    AuthResponse
)
from .model_server import ModelServer, ModelPool
from .cache import initialize_caches, get_prediction_cache, get_model_cache
from .middleware import (
    LoggingMiddleware, CORSMiddleware as CustomCORSMiddleware, RateLimitMiddleware,
    SecurityHeadersMiddleware, get_current_user, get_optional_user
)
from .exceptions import (
    BaseAPIException, ValidationError, InferenceError, ServiceUnavailableError,
    api_exception_handler, validation_exception_handler, http_exception_handler_custom,
    general_exception_handler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration in seconds', ['method', 'endpoint'])
INFERENCE_DURATION = Histogram('model_inference_duration_seconds', 'Model inference duration in seconds')
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Number of active connections')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
MODEL_ERRORS = Counter('model_errors_total', 'Total model errors', ['error_type'])

# Global state
model_server: Optional[ModelServer] = None
model_pool: Optional[ModelPool] = None
redis_client: Optional[redis.Redis] = None
startup_time = datetime.now()
websocket_connections: Dict[str, WebSocket] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Time-Series Transformer API...")
    
    try:
        # Initialize caches
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        
        initialize_caches(redis_host, redis_port, redis_db)
        
        # Initialize Redis client for middleware
        global redis_client
        try:
            redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError:
            logger.warning("Redis connection failed. Some features will be disabled.")
            redis_client = None
        
        # Initialize model server
        model_path = os.getenv('MODEL_PATH', 'models/best_model.pt')
        scaler_path = os.getenv('SCALER_PATH', 'models/scaler.pkl')
        device = os.getenv('DEVICE', 'auto')
        
        global model_server, model_pool
        
        # Check if model pool is requested
        pool_size = int(os.getenv('MODEL_POOL_SIZE', '1'))
        
        if pool_size > 1:
            model_configs = [{
                'model_path': model_path,
                'scaler_path': scaler_path,
                'device': device
            }]
            model_pool = ModelPool(model_configs, pool_size)
            logger.info(f"Initialized model pool with {pool_size} instances")
        else:
            model_server = ModelServer(model_path, scaler_path, device)
            logger.info("Initialized single model server instance")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")
    
    # Close WebSocket connections
    for conn_id, websocket in websocket_connections.items():
        try:
            await websocket.close()
        except:
            pass
    
    # Close Redis connection
    if redis_client:
        redis_client.close()
    
    logger.info("API shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Time-Series Transformer API",
    version="1.0.0",
    description="Production-ready API for stock price prediction using transformer models",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    redis_client=None,  # Will be set after Redis initialization
    default_rate_limit=100,
    time_window=60
)
app.add_middleware(
    CustomCORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600
)
app.add_middleware(LoggingMiddleware)

# Add exception handlers
app.add_exception_handler(BaseAPIException, api_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


def get_model_instance() -> ModelServer:
    """Get model server instance"""
    if model_pool:
        return model_pool.get_instance()
    elif model_server:
        return model_server
    else:
        raise ServiceUnavailableError("Model server not available")


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time metrics"""
    start_time = time.time()
    
    # Track active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        process_time = time.time() - start_time
        REQUEST_DURATION.labels(method=request.method, endpoint=request.url.path).observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
        
    finally:
        ACTIVE_CONNECTIONS.dec()


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    user: Optional[AuthResponse] = Depends(get_optional_user)
):
    """Generate prediction for a single ticker"""
    request_id = str(uuid.uuid4())
    
    try:
        # Check cache first
        cache = get_prediction_cache()
        if cache:
            cache_key = cache.generate_cache_key(request)
            cached_response = cache.get(cache_key)
            
            if cached_response:
                CACHE_HITS.labels(cache_type="prediction").inc()
                return cached_response
            else:
                CACHE_MISSES.labels(cache_type="prediction").inc()
        
        # Get model instance
        model_instance = get_model_instance()
        
        # Run inference
        start_time = time.time()
        response = model_instance.predict(request, request_id)
        inference_time = time.time() - start_time
        
        INFERENCE_DURATION.observe(inference_time)
        
        # Cache response
        if cache:
            background_tasks.add_task(cache.set, cache_key, response, ttl=300)
        
        return response
        
    except Exception as e:
        MODEL_ERRORS.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction failed for {request.ticker}: {e}")
        
        if isinstance(e, (ValueError, TypeError)):
            raise ValidationError(str(e), request_id=request_id)
        else:
            raise InferenceError(f"Prediction failed: {str(e)}", request_id=request_id)


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user: Optional[AuthResponse] = Depends(get_optional_user)
):
    """Generate predictions for multiple tickers"""
    request_id = str(uuid.uuid4())
    batch_start_time = time.time()
    
    try:
        # Process requests in parallel
        tasks = []
        cache = get_prediction_cache()
        
        for i, pred_request in enumerate(request.requests):
            task_id = f"{request_id}_{i}"
            
            # Check cache
            cached_response = None
            cache_key = None
            
            if cache:
                cache_key = cache.generate_cache_key(pred_request)
                cached_response = cache.get(cache_key)
                
                if cached_response:
                    CACHE_HITS.labels(cache_type="prediction").inc()
                    tasks.append(asyncio.create_task(asyncio.coroutine(lambda r=cached_response: r)()))
                    continue
                else:
                    CACHE_MISSES.labels(cache_type="prediction").inc()
            
            # Create prediction task
            async def predict_single(req, tid):
                model_instance = get_model_instance()
                result = model_instance.predict(req, tid)
                
                # Cache result
                if cache and cache_key:
                    background_tasks.add_task(cache.set, cache_key, result, ttl=300)
                
                return result
            
            task = asyncio.create_task(predict_single(pred_request, task_id))
            tasks.append(task)
        
        # Wait for all predictions to complete
        predictions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_predictions = []
        errors = []
        
        for i, result in enumerate(predictions):
            if isinstance(result, Exception):
                MODEL_ERRORS.labels(error_type=type(result).__name__).inc()
                errors.append({
                    "index": i,
                    "ticker": request.requests[i].ticker,
                    "error": str(result)
                })
            else:
                successful_predictions.append(result)
        
        batch_time = time.time() - batch_start_time
        
        # Create batch metadata
        batch_metadata = {
            "batch_id": request_id,
            "total_requests": len(request.requests),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(errors),
            "batch_processing_time_ms": round(batch_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "errors": errors if errors else None
        }
        
        return BatchPredictionResponse(
            predictions=successful_predictions,
            batch_metadata=batch_metadata
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise InferenceError(f"Batch prediction failed: {str(e)}", request_id=request_id)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        uptime = (datetime.now() - startup_time).total_seconds()
        
        # Check model status
        model_status = {}
        if model_pool:
            pool_health = model_pool.health_check()
            model_status = {f"model_{k}": "healthy" if v else "unhealthy" for k, v in pool_health.items()}
        elif model_server:
            model_status["model_primary"] = "healthy" if model_server.health_check() else "unhealthy"
        
        # Check dependencies
        dependencies = {}
        
        # Redis health
        if redis_client:
            try:
                redis_client.ping()
                dependencies["redis"] = "healthy"
            except:
                dependencies["redis"] = "unhealthy"
        else:
            dependencies["redis"] = "disabled"
        
        # Cache health
        cache = get_prediction_cache()
        if cache:
            dependencies["prediction_cache"] = "healthy" if cache.health_check() else "unhealthy"
        else:
            dependencies["prediction_cache"] = "disabled"
        
        # Determine overall status
        all_healthy = (
            all(status == "healthy" for status in model_status.values()) and
            all(status in ["healthy", "disabled"] for status in dependencies.values())
        )
        
        overall_status = "healthy" if all_healthy else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="1.0.0",
            uptime_seconds=uptime,
            model_status=model_status,
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            uptime_seconds=0,
            model_status={"error": str(e)},
            dependencies={}
        )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check for Kubernetes"""
    try:
        # Quick model check
        if model_pool:
            health_checks = model_pool.health_check()
            if not any(health_checks.values()):
                raise HTTPException(status_code=503, detail="No healthy model instances")
        elif model_server:
            if not model_server.health_check():
                raise HTTPException(status_code=503, detail="Model server unhealthy")
        else:
            raise HTTPException(status_code=503, detail="No model server available")
        
        return {"status": "ready"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information"""
    try:
        if model_pool:
            # Return info from first instance
            instance = model_pool.get_instance()
            model_info = instance.get_model_info()
            pool_info = model_pool.get_pool_info()
            model_info.update({"pool_info": pool_info})
        elif model_server:
            model_info = model_server.get_model_info()
        else:
            raise ServiceUnavailableError("Model server not available")
        
        return ModelInfoResponse(
            model_version=model_info.get("model_version", "unknown"),
            architecture="transformer",
            parameters=model_info.get("parameters", 0),
            device=model_info.get("device", "unknown"),
            loaded_at=datetime.fromisoformat(model_info["loaded_at"]) if model_info.get("loaded_at") else startup_time,
            training_metrics={}  # TODO: Load from model metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise ServiceUnavailableError(f"Failed to get model info: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics_summary():
    """Get metrics summary"""
    try:
        # Get cache stats
        cache_stats = {"hit_rate": 0.0}
        cache = get_prediction_cache()
        if cache:
            cache_stats = cache.get_stats()
        
        return MetricsResponse(
            active_connections=int(ACTIVE_CONNECTIONS._value._value),
            total_requests=int(REQUEST_COUNT._value.sum()),
            cache_hit_rate=cache_stats.get("hit_rate", 0.0) * 100,
            avg_inference_time_ms=0.0,  # TODO: Calculate from histogram
            error_rate=0.0  # TODO: Calculate error rate
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@app.websocket("/ws/stream/{ticker}")
async def websocket_endpoint(websocket: WebSocket, ticker: str):
    """WebSocket endpoint for streaming predictions"""
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    websocket_connections[connection_id] = websocket
    
    try:
        logger.info(f"WebSocket connection established: {connection_id} for {ticker}")
        
        # Send welcome message
        welcome_msg = WebSocketMessage(
            type="status",
            data={"message": f"Connected to {ticker} stream", "connection_id": connection_id},
            timestamp=datetime.now(),
            request_id=connection_id
        )
        await websocket.send_text(welcome_msg.json())
        
        # Handle incoming messages
        while True:
            try:
                # Wait for client message or timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                # Process streaming request
                if message.get("type") == "start_stream":
                    streaming_request = StreamingRequest(**message.get("data", {}))
                    await handle_streaming_predictions(websocket, streaming_request, connection_id)
                
            except asyncio.TimeoutError:
                # Send keep-alive message
                keepalive_msg = WebSocketMessage(
                    type="keepalive",
                    data={"timestamp": datetime.now().isoformat()},
                    timestamp=datetime.now(),
                    request_id=connection_id
                )
                await websocket.send_text(keepalive_msg.json())
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        error_msg = WebSocketMessage(
            type="error",
            data={"error": str(e)},
            timestamp=datetime.now(),
            request_id=connection_id
        )
        try:
            await websocket.send_text(error_msg.json())
        except:
            pass
    finally:
        websocket_connections.pop(connection_id, None)


async def handle_streaming_predictions(websocket: WebSocket, request: StreamingRequest, connection_id: str):
    """Handle streaming prediction requests"""
    try:
        while True:
            # TODO: Implement actual streaming logic with live data
            # For now, send periodic mock predictions
            
            prediction_msg = WebSocketMessage(
                type="prediction",
                data={
                    "ticker": request.ticker,
                    "prediction": [1.0, 2.0, 3.0, 4.0, 5.0],  # Mock data
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.95
                },
                timestamp=datetime.now(),
                request_id=connection_id
            )
            
            await websocket.send_text(prediction_msg.json())
            await asyncio.sleep(request.update_interval)
            
    except WebSocketDisconnect:
        logger.info(f"Streaming stopped for connection: {connection_id}")
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        raise


# Custom OpenAPI documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom examples
    if "paths" in openapi_schema:
        # Add example for prediction endpoint
        if "/predict" in openapi_schema["paths"]:
            openapi_schema["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["example"] = {
                "ticker": "AAPL",
                "features": [[1.0] * 7] * 60,
                "horizon": 5
            }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", "1"))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False
    )