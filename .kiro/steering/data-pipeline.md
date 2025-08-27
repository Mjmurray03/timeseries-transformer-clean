# Data Pipeline Standards
---
inclusion: fileMatch
fileMatchPattern: "src/data/**/*.py"
priority: 2
---

## Data Collection Standards

### API Rate Limiting
```python
from ratelimit import limits, sleep_and_retry

class DataCollector:
    @sleep_and_retry
    @limits(calls=5, period=1)  # 5 calls per second
    def fetch_data(self, ticker):
        """Rate-limited data fetching"""
        return yf.download(ticker, progress=False)
```

### Data Validation Rules
```python
VALIDATION_RULES = {
    "min_trading_days": 252,  # 1 year minimum
    "max_missing_ratio": 0.05,  # Max 5% missing data
    "outlier_threshold": 10,  # 10 standard deviations
    "volume_min": 100000,  # Minimum daily volume
}

def validate_stock_data(df: pd.DataFrame) -> bool:
    """Comprehensive data validation"""
    checks = [
        len(df) >= VALIDATION_RULES["min_trading_days"],
        df.isnull().sum().sum() / df.size < VALIDATION_RULES["max_missing_ratio"],
        ~any(abs(df['Returns']) > VALIDATION_RULES["outlier_threshold"] * df['Returns'].std()),
        df['Volume'].min() > VALIDATION_RULES["volume_min"]
    ]
    return all(checks)
```

### Data Schema Enforcement
```python
from pydantic import BaseModel, validator
from datetime import datetime

class StockData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    @validator('high')
    def high_greater_than_low(cls, v, values):
        if v < values.get('low', 0):
            raise ValueError('High must be >= Low')
        return v
    
    @validator('volume')
    def volume_positive(cls, v):
        if v < 0:
            raise ValueError('Volume must be positive')
        return v
```

## Feature Engineering Standards

### Technical Indicators Implementation
```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI calculation with proper initialization
    
    Args:
        prices: Close prices series
        period: RSI period (default 14)
    
    Returns:
        RSI values [0, 100]
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle initialization period
    rsi[:period] = 50  # Neutral RSI for initial period
    
    return rsi
```

### Feature Scaling Patterns
```python
class FeatureScaler:
    """Per-stock scaling to prevent data leakage"""
    
    def __init__(self):
        self.scalers = {}
    
    def fit_transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Fit scalers per stock and transform"""
        scaled_data = {}
        
        for ticker, df in data.items():
            # Create scaler for this stock
            scaler = StandardScaler()
            
            # Fit only on training portion (first 70%)
            train_size = int(len(df) * 0.7)
            scaler.fit(df.iloc[:train_size])
            
            # Transform all data
            scaled_data[ticker] = scaler.transform(df)
            
            # Save scaler for inference
            self.scalers[ticker] = scaler
            
        return scaled_data
    
    def save(self, path: str):
        """Save scalers for production use"""
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
```

### Sequence Generation Best Practices
```python
def create_sequences(
    data: np.ndarray,
    window: int = 60,
    horizon: int = 5,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping sequences with proper train/test separation
    
    Args:
        data: Feature matrix (time, features)
        window: Historical window size
        horizon: Prediction horizon
        stride: Step size between sequences
    
    Returns:
        X: Input sequences (n_sequences, window, features)
        y: Target sequences (n_sequences, horizon)
    """
    X, y = [], []
    
    for i in range(0, len(data) - window - horizon + 1, stride):
        # Input sequence
        X.append(data[i:i + window])
        
        # Target (only closing prices)
        y.append(data[i + window:i + window + horizon, 3])  # Close price column
    
    return np.array(X), np.array(y)
```

## Data Quality Monitoring

### Missing Data Handling
```python
class MissingDataHandler:
    """Sophisticated missing data imputation"""
    
    STRATEGIES = {
        'forward_fill': lambda s: s.fillna(method='ffill'),
        'interpolate': lambda s: s.interpolate(method='linear'),
        'mean_fill': lambda s: s.fillna(s.mean()),
        'drop': lambda s: s.dropna()
    }
    
    def handle_missing(self, df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
        """Apply missing data strategy"""
        if df.isnull().sum().sum() == 0:
            return df
        
        # Log missing data stats
        missing_stats = df.isnull().sum()
        logger.warning(f"Missing data found: {missing_stats.to_dict()}")
        
        # Apply strategy
        handler = self.STRATEGIES[strategy]
        df_filled = df.apply(handler)
        
        # Validate no missing data remains
        assert df_filled.isnull().sum().sum() == 0, "Missing data remains after handling"
        
        return df_filled
```

### Outlier Detection and Treatment
```python
def detect_outliers(data: pd.Series, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers using various methods
    
    Methods:
        - iqr: Interquartile range
        - zscore: Z-score method
        - isolation: Isolation Forest
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > 3
        
    elif method == 'isolation':
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.01)
        outliers = clf.fit_predict(data.values.reshape(-1, 1)) == -1
    
    return outliers

def handle_outliers(data: pd.Series, outliers: pd.Series, method: str = 'clip') -> pd.Series:
    """
    Handle detected outliers
    
    Methods:
        - clip: Cap at percentiles
        - remove: Drop outliers
        - transform: Log transformation
    """
    if method == 'clip':
        lower = data.quantile(0.01)
        upper = data.quantile(0.99)
        return data.clip(lower=lower, upper=upper)
    
    elif method == 'remove':
        return data[~outliers]
    
    elif method == 'transform':
        # Log transformation for positive values
        if (data > 0).all():
            return np.log1p(data)
        return data
```

## Data Storage Patterns

### Efficient Storage Formats
```python
class DataStorage:
    """Optimized data storage management"""
    
    @staticmethod
    def save_timeseries(data: pd.DataFrame, path: str, compress: bool = True):
        """Save time-series data efficiently"""
        if path.endswith('.parquet'):
            # Parquet for columnar storage
            data.to_parquet(path, compression='snappy' if compress else None)
        
        elif path.endswith('.h5'):
            # HDF5 for large datasets
            data.to_hdf(path, key='data', mode='w', complevel=9 if compress else 0)
        
        elif path.endswith('.feather'):
            # Feather for fast I/O
            data.to_feather(path, compression='lz4' if compress else None)
        
        else:
            # CSV as fallback
            data.to_csv(path, compression='gzip' if compress else None)
    
    @staticmethod
    def load_timeseries(path: str) -> pd.DataFrame:
        """Load time-series data with appropriate method"""
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.h5'):
            return pd.read_hdf(path)
        elif path.endswith('.feather'):
            return pd.read_feather(path)
        else:
            return pd.read_csv(path, parse_dates=['Date'], index_col='Date')
```

### Data Versioning
```python
import hashlib
from datetime import datetime

class DataVersioning:
    """Track data versions and lineage"""
    
    def __init__(self):
        self.versions = {}
    
    def create_version(self, data: pd.DataFrame, metadata: dict) -> str:
        """Create versioned snapshot of data"""
        # Generate version hash
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(data).values).hexdigest()[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"v_{timestamp}_{data_hash}"
        
        # Store version info
        self.versions[version_id] = {
            'shape': data.shape,
            'columns': list(data.columns),
            'date_range': (data.index.min(), data.index.max()),
            'metadata': metadata,
            'created_at': datetime.now()
        }
        
        # Save data
        version_path = f"data/versions/{version_id}.parquet"
        data.to_parquet(version_path)
        
        return version_id
    
    def load_version(self, version_id: str) -> pd.DataFrame:
        """Load specific data version"""
        version_path = f"data/versions/{version_id}.parquet"
        return pd.read_parquet(version_path)
```

## Real-time Data Pipeline

### Streaming Data Handler
```python
import asyncio
from typing import AsyncGenerator

class StreamingDataPipeline:
    """Handle real-time market data streams"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    async def stream_market_data(self, tickers: List[str]) -> AsyncGenerator:
        """Stream real-time market data"""
        async with aiohttp.ClientSession() as session:
            while True:
                for ticker in tickers:
                    data = await self.fetch_real_time(session, ticker)
                    yield data
                    
                    # Buffer management
                    self.buffer.append(data)
                    if len(self.buffer) >= self.buffer_size:
                        await self.flush_buffer()
                
                await asyncio.sleep(1)  # Rate limiting
    
    async def fetch_real_time(self, session, ticker):
        """Fetch real-time quote"""
        # Implementation depends on data provider
        pass
    
    async def flush_buffer(self):
        """Process buffered data"""
        df = pd.DataFrame(self.buffer)
        
        # Feature engineering on batch
        features = self.engineer_features(df)
        
        # Make predictions
        predictions = await self.predict_batch(features)
        
        # Clear buffer
        self.buffer = []
        
        return predictions
```

## Data Pipeline Testing

### Unit Tests for Data Functions
```python
import pytest

class TestDataPipeline:
    """Comprehensive data pipeline testing"""
    
    def test_sequence_generation(self):
        """Test sequence generation logic"""
        # Create sample data
        data = np.random.randn(100, 7)
        
        # Generate sequences
        X, y = create_sequences(data, window=60, horizon=5)
        
        # Assertions
        assert X.shape == (36, 60, 7)
        assert y.shape == (36, 5)
        assert X[0, -1, 3] == data[59, 3]  # Last input matches
        assert y[0, 0] == data[60, 3]  # First target matches
    
    def test_missing_data_handling(self):
        """Test missing data imputation"""
        # Create data with missing values
        data = pd.Series([1, 2, np.nan, 4, 5])
        
        # Handle missing data
        handler = MissingDataHandler()
        filled = handler.handle_missing(data, strategy='interpolate')
        
        # Assertions
        assert filled.isnull().sum() == 0
        assert filled.iloc[2] == 3.0  # Interpolated value
    
    def test_outlier_detection(self):
        """Test outlier detection methods"""
        # Create data with outlier
        data = pd.Series([1, 2, 3, 4, 100, 5, 6])
        
        # Detect outliers
        outliers = detect_outliers(data, method='zscore')
        
        # Assertions
        assert outliers.sum() == 1
        assert outliers.iloc[4] == True  # 100 is outlier
```

### Integration Tests
```python
class TestDataIntegration:
    """End-to-end data pipeline tests"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data"""
        dates = pd.date_range('2023-01-01', '2024-01-01')
        return pd.DataFrame({
            'Open': np.random.randn(len(dates)) * 10 + 100,
            'High': np.random.randn(len(dates)) * 10 + 105,
            'Low': np.random.randn(len(dates)) * 10 + 95,
            'Close': np.random.randn(len(dates)) * 10 + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_full_pipeline(self, sample_data):
        """Test complete data pipeline"""
        # 1. Validation
        assert validate_stock_data(sample_data)
        
        # 2. Feature engineering
        features = engineer_features(sample_data)
        assert 'RSI' in features.columns
        assert 'MACD' in features.columns
        
        # 3. Scaling
        scaler = FeatureScaler()
        scaled = scaler.fit_transform({'TEST': features})
        assert abs(scaled['TEST'].mean()) < 0.1  # Near zero mean
        assert abs(scaled['TEST'].std() - 1.0) < 0.1  # Near unit variance
        
        # 4. Sequence generation
        X, y = create_sequences(scaled['TEST'])
        assert len(X) > 0
```