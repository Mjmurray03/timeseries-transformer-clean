from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import redis
import json
import hashlib
from pathlib import Path

# Initialize FastAPI
app = FastAPI(
    title="TimeSeries Transformer API",
    version="1.0.0",
    description="Production API for stock price predictions"
)

# CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# REQUEST/RESPONSE MODELS
class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    features: List[List[float]] = Field(..., description="60 days x 10 features")
    horizon: int = Field(3, ge=1, le=10, description="Prediction horizon in days")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 60:
            raise ValueError("Must provide exactly 60 days of features")
        if any(len(day) != 10 for day in v):
            raise ValueError("Each day must have exactly 10 features")
        return v
    
    @validator('ticker')
    def validate_ticker(cls, v):
        allowed = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'NVDA', 'TSLA', 'NFLX']
        if v not in allowed:
            raise ValueError(f"Ticker must be one of {allowed}")
        return v

class PredictionResponse(BaseModel):
    ticker: str
    predictions: List[float] = Field(..., description="Predicted prices for horizon")
    confidence_intervals: Dict[str, List[float]] = Field(..., description="CI bounds")
    timestamp: datetime
    model_version: str
    cache_hit: bool = False

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str = Field(..., description="YYYY-MM-DD format")
    end_date: str = Field(..., description="YYYY-MM-DD format")
    initial_capital: float = Field(100000, gt=0)
    strategy_params: Dict = Field(
        default={
            "return_threshold": 0.02,
            "confidence_threshold": 0.7,
            "max_positions": 5
        }
    )

class BacktestResponse(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    period_days: int

class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    parameters: int
    training_date: str
    supported_tickers: List[str]
    performance_metrics: Dict[str, float]
# GLOBAL MODEL LOADING
class ModelManager:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models on startup"""
        model_dir = Path("models")
        scaler_dir = Path("scalers")
        
        for model_path in model_dir.glob("*_best.pt"):
            ticker = model_path.stem.split("_")[0]
            
            # Load model
            model = self._create_model()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            model.to(self.device)
            self.models[ticker] = model
            
            # Load scaler
            scaler_path = scaler_dir / f"scaler_{ticker}.json"
            if scaler_path.exists():
                with open(scaler_path) as f:
                    self.scalers[ticker] = json.load(f)
    
    def _create_model(self):
        """Create model architecture - imported from src.models"""
        from src.models.timeseries_transformer import TimeSeriesTransformer
        return TimeSeriesTransformer(
            input_dim=10,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            max_seq_length=60,
            output_dim=3,
            forecast_horizon=5,
            use_attention_pooling=True
        )
    
    @torch.no_grad()
    def predict(self, ticker: str, features: np.ndarray) -> Dict:
        """Generate prediction with model"""
        if ticker not in self.models:
            raise ValueError(f"No model available for {ticker}")
        
        model = self.models[ticker]
        scaler = self.scalers[ticker]
        
        # Standardize features
        features_scaled = (features - scaler['feat_mean']) / scaler['feat_std']
        
        # Convert to tensor
        x = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        output = model(x)
        pred_scaled = output.cpu().numpy()[0]
        
        # De-standardize predictions
        pred_dollars = pred_scaled * scaler['tgt_std'] + scaler['tgt_mean']
        
        # Calculate confidence intervals (using dropout uncertainty)
        model.train()  # Enable dropout
        predictions = []
        for _ in range(100):
            out = model(x).cpu().numpy()[0]
            predictions.append(out * scaler['tgt_std'] + scaler['tgt_mean'])
        model.eval()
        
        predictions = np.array(predictions)
        ci_lower = np.percentile(predictions, 5, axis=0)
        ci_upper = np.percentile(predictions, 95, axis=0)
        
        return {
            'predictions': pred_dollars.tolist(),
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist()
        }
# CACHE MANAGER
class CacheManager:
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis.ping()
            self.enabled = True
        except:
            self.enabled = False
    
    def get_cache_key(self, request_dict: dict) -> str:
        """Generate deterministic cache key"""
        content = json.dumps(request_dict, sort_keys=True)
        return f"prediction:{hashlib.sha256(content.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        try:
            cached = self.redis.get(key)
            return json.loads(cached) if cached else None
        except:
            return None
    
    def set(self, key: str, value: Dict, expire: int = 3600):
        if not self.enabled:
            return
        try:
            self.redis.setex(key, expire, json.dumps(value))
        except:
            pass

# INITIALIZE MANAGERS
model_manager = ModelManager()
cache_manager = CacheManager()

# ENDPOINTS

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(model_manager.models.keys()),
        "cache_enabled": cache_manager.enabled
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate price predictions for given features"""
    
    # Check cache
    cache_key = cache_manager.get_cache_key(request.dict())
    cached = cache_manager.get(cache_key)
    if cached:
        return PredictionResponse(**cached, cache_hit=True)
    
    try:
        # Generate prediction
        features = np.array(request.features, dtype=np.float32)
        result = model_manager.predict(request.ticker, features)
        
        # Build response
        response_data = {
            "ticker": request.ticker,
            "predictions": result['predictions'],
            "confidence_intervals": {
                "lower": result['ci_lower'],
                "upper": result['ci_upper']
            },
            "timestamp": datetime.now(),
            "model_version": "1.0.0",
            "cache_hit": False
        }
        
        # Cache result
        cache_manager.set(cache_key, response_data)
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtesting simulation"""
    
    try:
        # Import backtesting engine
        from src.backtesting.backtest_engine import BacktestEngine
        
        # Load historical data
        data_path = f"data/raw/{request.ticker}.parquet"
        if not Path(data_path).exists():
            raise HTTPException(status_code=404, detail=f"No data for {request.ticker}")
        
        # Run backtest (simplified for API)
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            **request.strategy_params
        )
        
        results = engine.run_quick_backtest(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            model=model_manager.models[request.ticker]
        )
        
        return BacktestResponse(**results)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about loaded models"""
    
    return ModelInfoResponse(
        model_version="1.0.0",
        architecture="TimeSeriesTransformer",
        parameters=464571,
        training_date="2024-08-27",
        supported_tickers=list(model_manager.models.keys()),
        performance_metrics={
            "avg_rmse": 0.268,
            "avg_sharpe": 1.2,
            "directional_accuracy": 0.57
        }
    )

# STARTUP/SHUTDOWN EVENTS

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print(f"API Started. Models loaded: {list(model_manager.models.keys())}")
    print(f"Cache enabled: {cache_manager.enabled}")
    print(f"Device: {model_manager.device}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if cache_manager.enabled:
        cache_manager.redis.close()

# ERROR HANDLERS

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {"error": str(exc)}, 400

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )