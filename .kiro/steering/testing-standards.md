# Testing Standards
---
inclusion: fileMatch
fileMatchPattern: "tests/**/*.py"
priority: 2
---

## Testing Philosophy

### Test Pyramid
```
         /\
        /  \  E2E Tests (5%)
       /    \
      /------\  Integration Tests (20%)
     /        \
    /----------\  Unit Tests (75%)
   /            \
```

### Coverage Requirements
- Overall: Minimum 80%
- Critical paths: Minimum 95%
- Data pipeline: Minimum 90%
- Model components: Minimum 85%
- API endpoints: Minimum 90%

## Unit Testing Standards

### Test Structure Pattern
```python
class TestComponentName:
    """Test suite for ComponentName"""
    
    @pytest.fixture
    def component(self):
        """Create component instance for testing"""
        return ComponentName(test_config)
    
    @pytest.fixture
    def mock_data(self):
        """Generate mock data for testing"""
        return create_mock_data()
    
    def test_initialization(self, component):
        """Test component initializes correctly"""
        assert component is not None
        assert component.config == test_config
    
    def test_happy_path(self, component, mock_data):
        """Test normal operation succeeds"""
        result = component.process(mock_data)
        assert result.shape == expected_shape
        assert result.dtype == expected_dtype
    
    def test_edge_cases(self, component):
        """Test boundary conditions"""
        # Empty input
        assert component.process([]) == []
        
        # Single element
        assert len(component.process([1])) == 1
        
        # Maximum size
        large_input = [1] * MAX_SIZE
        assert component.process(large_input) is not None
    
    def test_error_handling(self, component):
        """Test error conditions raise appropriately"""
        with pytest.raises(ValueError):
            component.process(None)
        
        with pytest.raises(TypeError):
            component.process("invalid")
```

### Fixture Best Practices
```python
@pytest.fixture(scope="session")
def trained_model():
    """Session-scoped expensive fixture"""
    model = load_model("test_model.pt")
    yield model
    cleanup_model(model)

@pytest.fixture
def mock_api_response(monkeypatch):
    """Mock external API calls"""
    def mock_get(*args, **kwargs):
        return MockResponse({"data": "test"})
    
    monkeypatch.setattr(requests, "get", mock_get)
    
@pytest.fixture(params=[10, 100, 1000])
def batch_sizes(request):
    """Parametrized fixture for testing different sizes"""
    return request.param
```

### Mocking Guidelines
```python
from unittest.mock import Mock, patch, MagicMock

class TestDataCollector:
    @patch('yfinance.download')
    def test_download_with_mock(self, mock_download):
        """Test with mocked external dependency"""
        # Setup mock
        mock_download.return_value = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        
        # Execute
        collector = DataCollector()
        data = collector.fetch('AAPL')
        
        # Assert
        mock_download.assert_called_once_with('AAPL', progress=False)
        assert len(data) == 3
    
    @patch.object(DataValidator, 'validate')
    def test_validation_called(self, mock_validate):
        """Test internal method calls"""
        mock_validate.return_value = True
        
        pipeline = DataPipeline()
        pipeline.process()
        
        mock_validate.assert_called()
```

## Integration Testing Standards

### Database Testing
```python
@pytest.mark.integration
class TestDatabaseIntegration:
    @pytest.fixture
    def test_db(self):
        """Create test database"""
        db = create_test_database()
        yield db
        cleanup_database(db)
    
    def test_data_persistence(self, test_db):
        """Test data saves and loads correctly"""
        # Save
        test_db.save("key", test_data)
        
        # Load
        loaded = test_db.load("key")
        
        # Verify
        assert loaded == test_data
```

### API Testing
```python
@pytest.mark.integration
class TestAPIIntegration:
    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.api.app import app
        return TestClient(app)
    
    def test_prediction_endpoint(self, client):
        """Test prediction API endpoint"""
        response = client.post(
            "/predict",
            json={"ticker": "AAPL", "features": [[1.0] * 7] * 60}
        )
        
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert len(response.json()["prediction"]) == 5
    
    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
```

## Performance Testing Standards

### Benchmark Tests
```python
@pytest.mark.performance
class TestPerformance:
    @pytest.mark.benchmark(group="inference")
    def test_inference_speed(self, benchmark, trained_model):
        """Benchmark model inference speed"""
        input_data = torch.randn(1, 60, 7)
        
        result = benchmark(trained_model.predict, input_data)
        
        # Assert performance requirements
        assert benchmark.stats['mean'] < 0.01  # 10ms
        assert benchmark.stats['stddev'] < 0.002  # Low variance
    
    @pytest.mark.benchmark(group="training")
    def test_training_throughput(self, benchmark):
        """Benchmark training throughput"""
        def training_step():
            loss = model(batch)
            loss.backward()
            optimizer.step()
        
        benchmark.pedantic(
            training_step,
            rounds=100,
            warmup_rounds=10
        )
```

### Memory Testing
```python
@pytest.mark.memory
class TestMemoryUsage:
    def test_memory_leak(self):
        """Test for memory leaks during training"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Run training loop
        for _ in range(100):
            model.train_step(batch)
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Check memory growth
        for stat in top_stats[:10]:
            assert stat.size < 100 * 1024 * 1024  # 100MB max
```

### Load Testing
```python
@pytest.mark.load
async def test_concurrent_requests():
    """Test API under load"""
    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/predict",
                json=test_payload
            ) as response:
                return await response.json()
    
    # Create concurrent requests
    tasks = [make_request() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    
    # Verify all succeeded
    assert all(r["status"] == "success" for r in results)
```

## ML-Specific Testing

### Model Testing
```python
class TestModelBehavior:
    def test_model_deterministic(self, model):
        """Test model produces consistent outputs"""
        torch.manual_seed(42)
        input_data = torch.randn(1, 60, 7)
        
        output1 = model(input_data)
        output2 = model(input_data)
        
        torch.testing.assert_close(output1, output2)
    
    def test_gradient_flow(self, model):
        """Test gradients flow through model"""
        input_data = torch.randn(1, 60, 7, requires_grad=True)
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        
        assert input_data.grad is not None
        assert not torch.isnan(input_data.grad).any()
        assert not torch.isinf(input_data.grad).any()
    
    def test_batch_consistency(self, model):
        """Test batch processing consistency"""
        single_input = torch.randn(1, 60, 7)
        batch_input = single_input.repeat(5, 1, 1)
        
        single_output = model(single_input)
        batch_output = model(batch_input)
        
        # First item in batch should match single
        torch.testing.assert_close(
            single_output,
            batch_output[0:1],
            rtol=1e-5,
            atol=1e-5
        )
```

### Data Testing
```python
class TestDataQuality:
    def test_no_future_leakage(self, dataset):
        """Test no future information leaks into past"""
        for i in range(len(dataset)):
            x, y = dataset[i]
            x_dates = x.index
            y_dates = y.index
            
            # Target dates must be after input dates
            assert y_dates.min() > x_dates.max()
    
    def test_data_normalization(self, normalized_data):
        """Test data is properly normalized"""
        mean = normalized_data.mean()
        std = normalized_data.std()
        
        assert abs(mean) < 0.01  # Near zero mean
        assert abs(std - 1.0) < 0.01  # Unit variance
    
    def test_sequence_alignment(self, sequences):
        """Test sequences are properly aligned"""
        for seq in sequences:
            assert seq.shape[0] == WINDOW_SIZE
            assert seq.shape[1] == NUM_FEATURES
            assert not np.isnan(seq).any()
```

## Test Organization

### Test Discovery Pattern
```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set test mode
    os.environ["TESTING"] = "true"
    
    yield
    
    # Cleanup
    cleanup_test_files()

# Custom markers
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
```

### Test Execution Strategy
```yaml
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests (fast)
    integration: Integration tests (slower)
    performance: Performance benchmarks
    smoke: Smoke tests for deployment
    gpu: Tests requiring GPU
    slow: Tests taking > 1 second

# Coverage settings
addopts = 
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
```

## Continuous Testing

### Pre-commit Tests
```python
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: Unit Tests
        entry: pytest -m unit
        language: system
        pass_filenames: false
        always_run: true
```

### CI Pipeline Tests
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Unit Tests
        run: pytest -m unit --cov

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Integration Tests
        run: pytest -m integration

  performance-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Run Performance Tests
        run: pytest -m performance --benchmark-only
```

## Test Data Management

### Fixtures and Factories
```python
# tests/factories.py
import factory
from factory import fuzzy

class StockDataFactory(factory.Factory):
    """Factory for generating test stock data"""
    class Meta:
        model = pd.DataFrame
    
    open = fuzzy.FuzzyDecimal(90, 110)
    high = factory.LazyAttribute(lambda o: o.open * 1.05)
    low = factory.LazyAttribute(lambda o: o.open * 0.95)
    close = fuzzy.FuzzyDecimal(95, 105)
    volume = fuzzy.FuzzyInteger(1000000, 10000000)

def create_test_sequence(length=60):
    """Create test sequence data"""
    return StockDataFactory.create_batch(length)
```

### Test Data Versioning
```python
# tests/data/README.md
## Test Data Version Control

Test data is versioned using DVC (Data Version Control):

```bash
# Track test data
dvc add tests/data/sample_stocks.parquet

# Push to remote storage
dvc push

# Pull specific version
dvc checkout v1.0.0
```

### Golden Data Tests
```python
class TestGoldenData:
    """Test against known good outputs"""
    
    def test_against_golden(self, model):
        """Compare model output to golden data"""
        golden_input = load_golden("input_v1.pt")
        golden_output = load_golden("output_v1.pt")
        
        actual_output = model(golden_input)
        
        torch.testing.assert_close(
            actual_output,
            golden_output,
            rtol=1e-4,
            atol=1e-4
        )
```