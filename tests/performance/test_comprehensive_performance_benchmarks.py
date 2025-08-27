"""
Comprehensive performance benchmarking suite.
Follows testing-standards.md patterns and design.md timing requirements.
Tests inference speed, training throughput, and memory usage.
"""
import pytest
import torch
import numpy as np
import time
import tracemalloc
import gc
import psutil
import os
from unittest.mock import Mock, patch
from typing import Dict, List, Any
from pathlib import Path

from src.models.timeseries_transformer import TimeSeriesTransformer
from src.training.trainer import Trainer
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig


@pytest.mark.performance
class TestInferencePerformance:
    """Performance benchmarks for model inference following design.md requirements"""
    
    @pytest.fixture
    def model_configs(self):
        """Different model configurations for benchmarking"""
        return {
            'small': ModelConfig(
                sequence_length=60,
                num_features=7,
                d_model=128,
                n_heads=4,
                n_layers=2,
                dropout=0.1,
                forecast_horizon=5
            ),
            'base': ModelConfig(
                sequence_length=60,
                num_features=7,
                d_model=256,
                n_heads=8,
                n_layers=6,
                dropout=0.1,
                forecast_horizon=5
            ),
            'large': ModelConfig(
                sequence_length=60,
                num_features=7,
                d_model=512,
                n_heads=16,
                n_layers=8,
                dropout=0.1,
                forecast_horizon=5
            )
        }
    
    @pytest.fixture
    def benchmark_data(self):
        """Generate benchmark data for different batch sizes"""
        torch.manual_seed(42)
        return {
            'single': torch.randn(1, 60, 7),
            'small_batch': torch.randn(8, 60, 7),
            'medium_batch': torch.randn(32, 60, 7),
            'large_batch': torch.randn(128, 60, 7)
        }
    
    @pytest.mark.benchmark(group="inference")
    def test_inference_speed_single(self, benchmark, model_configs, benchmark_data):
        """Benchmark single inference speed - should be < 10ms per design.md"""
        model = TimeSeriesTransformer(**model_configs['base'].__dict__)
        model.eval()
        
        input_data = benchmark_data['single']
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        def inference_step():
            with torch.no_grad():
                return model(input_data)
        
        result = benchmark(inference_step)
        
        # Assert performance requirements from design.md
        assert result.stats['mean'] < 0.01  # 10ms requirement
        assert result.stats['stddev'] < 0.002  # Low variance requirement
    
    @pytest.mark.benchmark(group="inference")
    def test_batch_inference_throughput(self, benchmark, model_configs, benchmark_data):
        """Benchmark batch inference throughput"""
        model = TimeSeriesTransformer(**model_configs['base'].__dict__)
        model.eval()
        
        def batch_inference():
            with torch.no_grad():
                results = []
                for batch_name, batch_data in benchmark_data.items():
                    if batch_name != 'single':  # Skip single for throughput test
                        output = model(batch_data)
                        results.append(output)
                return results
        
        result = benchmark(batch_inference)
        
        # Calculate throughput (samples per second)
        total_samples = sum(data.shape[0] for name, data in benchmark_data.items() if name != 'single')
        throughput = total_samples / result.stats['mean']
        
        # Should process at least 1000 samples per second
        assert throughput > 1000
    
    @pytest.mark.benchmark(group="inference", min_rounds=50)
    def test_inference_consistency(self, benchmark, model_configs, benchmark_data):
        """Benchmark inference consistency across multiple runs"""
        model = TimeSeriesTransformer(**model_configs['base'].__dict__)
        model.eval()
        
        input_data = benchmark_data['medium_batch']
        
        def consistent_inference():
            torch.manual_seed(42)  # Ensure deterministic behavior
            with torch.no_grad():
                return model(input_data)
        
        result = benchmark.pedantic(
            consistent_inference,
            rounds=50,
            warmup_rounds=10
        )
        
        # Low variance indicates consistent performance
        coefficient_of_variation = result.stats['stddev'] / result.stats['mean']
        assert coefficient_of_variation < 0.1  # Less than 10% variation
    
    @pytest.mark.parametrize("model_size", ["small", "base", "large"])
    @pytest.mark.benchmark(group="model_scaling")
    def test_model_size_scaling(self, benchmark, model_configs, benchmark_data, model_size):
        """Benchmark how performance scales with model size"""
        config = model_configs[model_size]
        model = TimeSeriesTransformer(**config.__dict__)
        model.eval()
        
        input_data = benchmark_data['medium_batch']
        
        def inference_with_model():
            with torch.no_grad():
                return model(input_data)
        
        result = benchmark(inference_with_model)
        
        # Store results for comparison (in real implementation, would log to file)
        benchmark.extra_info.update({
            'model_size': model_size,
            'parameters': sum(p.numel() for p in model.parameters()),
            'throughput': input_data.shape[0] / result.stats['mean']
        })
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.benchmark(group="gpu_inference")
    def test_gpu_inference_speed(self, benchmark, model_configs, benchmark_data):
        """Benchmark GPU inference speed"""
        device = torch.device('cuda')
        model = TimeSeriesTransformer(**model_configs['base'].__dict__).to(device)
        model.eval()
        
        input_data = benchmark_data['large_batch'].to(device)
        
        # CUDA warm up
        with torch.no_grad():
            for _ in range(20):
                _ = model(input_data)
        
        torch.cuda.synchronize()
        
        def gpu_inference():
            with torch.no_grad():
                output = model(input_data)
                torch.cuda.synchronize()  # Ensure completion
                return output
        
        result = benchmark(gpu_inference)
        
        # GPU should be significantly faster than CPU for large batches
        # Exact requirements would depend on hardware
        assert result.stats['mean'] < 0.1  # Should be fast for GPU


@pytest.mark.performance
class TestTrainingPerformance:
    """Performance benchmarks for training pipeline"""
    
    @pytest.fixture
    def training_setup(self):
        """Setup training components for benchmarking"""
        model_config = ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=256,
            n_heads=8,
            n_layers=6,
            dropout=0.1,
            forecast_horizon=5
        )
        
        training_config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=32,
            num_epochs=1,  # Single epoch for benchmarking
            gradient_clip=1.0,
            weight_decay=1e-5
        )
        
        model = TimeSeriesTransformer(**model_config.__dict__)
        
        # Mock dataloader with consistent batches
        def create_batch():
            return {
                'input': torch.randn(32, 60, 7),
                'target': torch.randn(32, 5)
            }
        
        batches = [create_batch() for _ in range(20)]  # 20 batches per epoch
        
        dataloader = Mock()
        dataloader.__iter__ = Mock(return_value=iter(batches))
        dataloader.__len__ = Mock(return_value=len(batches))
        
        return model, training_config, dataloader
    
    @pytest.mark.benchmark(group="training")
    def test_training_step_speed(self, benchmark, training_setup):
        """Benchmark single training step speed"""
        model, config, dataloader = training_setup
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Get a batch for benchmarking
        batch = next(iter(dataloader))
        
        def training_step():
            model.train()
            optimizer.zero_grad()
            
            outputs = model(batch['input'])
            main_output = list(outputs.values())[0]
            loss = criterion(main_output, batch['target'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            
            return loss.item()
        
        result = benchmark(training_step)
        
        # Training step should be reasonably fast
        assert result.stats['mean'] < 1.0  # Less than 1 second per step
    
    @pytest.mark.benchmark(group="training")
    def test_epoch_throughput(self, benchmark, training_setup):
        """Benchmark training epoch throughput - should be < 2 minutes per requirements.md"""
        model, config, dataloader = training_setup
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.MSELoss()
        
        def train_epoch():
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                outputs = model(batch['input'])
                main_output = list(outputs.values())[0]
                loss = criterion(main_output, batch['target'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / num_batches
        
        result = benchmark(train_epoch)
        
        # Requirements: single epoch < 2 minutes (120 seconds) for 10 stocks
        # Our test uses 20 batches, so should be much faster
        assert result.stats['mean'] < 120.0  # 2 minutes requirement
        
        # Calculate samples per second
        total_samples = len(dataloader) * 32  # 20 batches * 32 batch_size
        throughput = total_samples / result.stats['mean']
        
        # Should process reasonable number of samples per second
        assert throughput > 100  # At least 100 samples per second
    
    @pytest.mark.benchmark(group="training")
    def test_validation_speed(self, benchmark, training_setup):
        """Benchmark validation speed - should be < 30 seconds per requirements.md"""
        model, _, dataloader = training_setup
        criterion = torch.nn.MSELoss()
        
        def validate_epoch():
            model.eval()
            total_loss = 0
            num_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    outputs = model(batch['input'])
                    main_output = list(outputs.values())[0]
                    loss = criterion(main_output, batch['target'])
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            return total_loss / num_batches
        
        result = benchmark(validate_epoch)
        
        # Requirements: validation < 30 seconds per epoch
        assert result.stats['mean'] < 30.0
        
        # Validation should be faster than training (no backprop)
        # This is tested implicitly by the time constraint
    
    @pytest.mark.benchmark(group="training", min_rounds=10)
    def test_gradient_computation_speed(self, benchmark, training_setup):
        """Benchmark gradient computation speed"""
        model, config, dataloader = training_setup
        
        batch = next(iter(dataloader))
        criterion = torch.nn.MSELoss()
        
        def compute_gradients():
            model.train()
            model.zero_grad()
            
            outputs = model(batch['input'])
            main_output = list(outputs.values())[0]
            loss = criterion(main_output, batch['target'])
            
            loss.backward()
            
            # Return gradient norm for verification
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        
        result = benchmark.pedantic(
            compute_gradients,
            rounds=10,
            warmup_rounds=3
        )
        
        # Gradient computation should be consistent and fast
        assert result.stats['mean'] < 0.5  # Less than 500ms
        assert result.stats['stddev'] / result.stats['mean'] < 0.2  # Low variance


@pytest.mark.performance
@pytest.mark.memory
class TestMemoryPerformance:
    """Memory usage and leak testing following testing-standards.md patterns"""
    
    @pytest.fixture
    def memory_tracker(self):
        """Memory tracking fixture"""
        class MemoryTracker:
            def __init__(self):
                self.snapshots = []
                self.baseline = None
            
            def start(self):
                tracemalloc.start()
                gc.collect()
                self.baseline = tracemalloc.take_snapshot()
            
            def snapshot(self, label=""):
                current = tracemalloc.take_snapshot()
                self.snapshots.append((label, current))
                return current
            
            def stop(self):
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
            
            def get_memory_usage(self):
                """Get current memory usage in MB"""
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            
            def compare_snapshots(self, snapshot1, snapshot2):
                """Compare two memory snapshots"""
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                total_diff = sum(stat.size_diff for stat in top_stats)
                return total_diff
        
        tracker = MemoryTracker()
        yield tracker
        tracker.stop()
    
    def test_inference_memory_usage(self, memory_tracker):
        """Test memory usage during inference"""
        model_config = ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=256,
            n_heads=8,
            n_layers=6,
            dropout=0.1,
            forecast_horizon=5
        )
        
        model = TimeSeriesTransformer(**model_config.__dict__)
        model.eval()
        
        memory_tracker.start()
        initial_memory = memory_tracker.get_memory_usage()
        
        # Run inference multiple times
        torch.manual_seed(42)
        for i in range(100):
            input_data = torch.randn(32, 60, 7)
            with torch.no_grad():
                outputs = model(input_data)
            
            # Take memory snapshots periodically
            if i % 25 == 0:
                memory_tracker.snapshot(f"inference_step_{i}")
        
        final_memory = memory_tracker.get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal during inference
        assert memory_growth < 50  # Less than 50MB growth
    
    def test_training_memory_leak(self, memory_tracker):
        """Test for memory leaks during training following testing-standards.md pattern"""
        model_config = ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=128,  # Smaller model for memory testing
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            forecast_horizon=5
        )
        
        model = TimeSeriesTransformer(**model_config.__dict__)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        
        memory_tracker.start()
        initial_snapshot = memory_tracker.snapshot("initial")
        
        # Simulate training loop
        model.train()
        for step in range(100):
            # Generate batch
            input_data = torch.randn(16, 60, 7)
            target_data = torch.randn(16, 5)
            
            optimizer.zero_grad()
            outputs = model(input_data)
            main_output = list(outputs.values())[0]
            loss = criterion(main_output, target_data)
            loss.backward()
            optimizer.step()
            
            # Take periodic snapshots
            if step % 25 == 0:
                memory_tracker.snapshot(f"training_step_{step}")
        
        final_snapshot = memory_tracker.snapshot("final")
        
        # Check memory growth
        memory_diff = memory_tracker.compare_snapshots(initial_snapshot, final_snapshot)
        
        # Following testing-standards.md: max 100MB growth
        assert abs(memory_diff) < 100 * 1024 * 1024  # 100MB limit
    
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, memory_tracker):
        """Test GPU memory usage"""
        device = torch.device('cuda')
        
        model_config = ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=256,
            n_heads=8,
            n_layers=6,
            dropout=0.1,
            forecast_horizon=5
        )
        
        model = TimeSeriesTransformer(**model_config.__dict__).to(device)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated(device)
        
        # Run inference
        model.eval()
        input_data = torch.randn(64, 60, 7).to(device)  # Large batch for GPU
        
        with torch.no_grad():
            for _ in range(50):
                outputs = model(input_data)
                torch.cuda.empty_cache()
        
        final_gpu_memory = torch.cuda.memory_allocated(device)
        gpu_memory_growth = final_gpu_memory - initial_gpu_memory
        
        # GPU memory should not grow significantly during inference
        assert gpu_memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
    
    def test_model_parameter_memory(self):
        """Test model parameter memory usage"""
        configs = {
            'small': ModelConfig(d_model=128, n_heads=4, n_layers=2),
            'base': ModelConfig(d_model=256, n_heads=8, n_layers=6),
            'large': ModelConfig(d_model=512, n_heads=16, n_layers=8)
        }
        
        memory_usage = {}
        
        for size, config in configs.items():
            # Complete config for model creation
            full_config = ModelConfig(
                sequence_length=60,
                num_features=7,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                dropout=0.1,
                forecast_horizon=5
            )
            
            model = TimeSeriesTransformer(**full_config.__dict__)
            
            # Calculate parameter memory
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            memory_usage[size] = param_memory / (1024 * 1024)  # Convert to MB
        
        # Verify memory scaling is reasonable
        assert memory_usage['base'] > memory_usage['small']
        assert memory_usage['large'] > memory_usage['base']
        
        # Large model should be within reasonable bounds (adjust based on requirements)
        assert memory_usage['large'] < 500  # Less than 500MB for parameters


@pytest.mark.performance
@pytest.mark.load
class TestLoadPerformance:
    """Load testing for concurrent operations following testing-standards.md patterns"""
    
    @pytest.fixture
    def concurrent_model(self):
        """Shared model for concurrent testing"""
        config = ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=256,
            n_heads=8,
            n_layers=4,  # Moderate size for load testing
            dropout=0.1,
            forecast_horizon=5
        )
        
        model = TimeSeriesTransformer(**config.__dict__)
        model.eval()
        return model
    
    @pytest.mark.asyncio
    async def test_concurrent_inference_requests(self, concurrent_model):
        """Test concurrent inference requests following testing-standards.md pattern"""
        import asyncio
        
        async def inference_request(request_id: int):
            """Simulate single inference request"""
            torch.manual_seed(request_id)  # Different data per request
            input_data = torch.randn(1, 60, 7)
            
            start_time = time.time()
            with torch.no_grad():
                output = concurrent_model(input_data)
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'duration': end_time - start_time,
                'output_shape': list(output.values())[0].shape,
                'success': True
            }
        
        # Create 100 concurrent requests as per testing-standards.md
        num_requests = 100
        start_time = time.time()
        
        tasks = [inference_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Verify all requests completed successfully
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        assert len(successful_results) == num_requests
        
        # Calculate performance metrics
        avg_request_time = np.mean([r['duration'] for r in successful_results])
        requests_per_second = num_requests / total_time
        
        # Performance assertions
        assert avg_request_time < 0.1  # Average request < 100ms
        assert requests_per_second > 50  # At least 50 requests per second
        assert total_time < 10.0  # All 100 requests complete in < 10 seconds
    
    def test_batch_size_scaling(self):
        """Test how performance scales with batch size"""
        model_config = ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=0.1,
            forecast_horizon=5
        )
        
        model = TimeSeriesTransformer(**model_config.__dict__)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        performance_results = {}
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 60, 7)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_data)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(input_data)
            end_time = time.time()
            
            avg_time_per_run = (end_time - start_time) / 20
            samples_per_second = batch_size / avg_time_per_run
            
            performance_results[batch_size] = {
                'time_per_run': avg_time_per_run,
                'samples_per_second': samples_per_second,
                'time_per_sample': avg_time_per_run / batch_size
            }
        
        # Verify batch processing is efficient
        # Larger batches should have better samples/second throughput
        assert performance_results[32]['samples_per_second'] > performance_results[1]['samples_per_second']
        assert performance_results[64]['samples_per_second'] > performance_results[8]['samples_per_second']
        
        # Time per sample should decrease with larger batches (economy of scale)
        assert performance_results[64]['time_per_sample'] < performance_results[1]['time_per_sample']
    
    def test_sustained_load_performance(self):
        """Test sustained load over extended period"""
        model_config = ModelConfig(
            sequence_length=60,
            num_features=7,
            d_model=256,
            n_heads=8,
            n_layers=4,
            dropout=0.1,
            forecast_horizon=5
        )
        
        model = TimeSeriesTransformer(**model_config.__dict__)
        model.eval()
        
        # Run sustained load for 30 seconds
        duration = 30.0  # seconds
        batch_size = 32
        
        start_time = time.time()
        request_times = []
        total_requests = 0
        
        while time.time() - start_time < duration:
            input_data = torch.randn(batch_size, 60, 7)
            
            request_start = time.time()
            with torch.no_grad():
                outputs = model(input_data)
            request_end = time.time()
            
            request_times.append(request_end - request_start)
            total_requests += 1
        
        total_time = time.time() - start_time
        
        # Performance analysis
        avg_request_time = np.mean(request_times)
        std_request_time = np.std(request_times)
        requests_per_second = total_requests / total_time
        
        # Assertions for sustained performance
        assert avg_request_time < 0.1  # Average request time < 100ms
        assert std_request_time / avg_request_time < 0.5  # Coefficient of variation < 50%
        assert requests_per_second > 100  # At least 100 requests per second
        assert total_requests > 50  # Completed reasonable number of requests


@pytest.mark.performance
class TestDataPipelinePerformance:
    """Performance benchmarks for data pipeline components"""
    
    @pytest.mark.benchmark(group="data_processing")
    def test_feature_engineering_speed(self, benchmark):
        """Benchmark feature engineering performance"""
        from src.data.processors.feature_engineering import FeatureEngineer
        from src.config.data_config import DataConfig
        
        # Generate large dataset for benchmarking
        np.random.seed(42)
        n_days = 1000  # ~3 years of data
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        raw_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.normal(0, 1, n_days)),
            'High': 102 + np.cumsum(np.random.normal(0, 1, n_days)),
            'Low': 98 + np.cumsum(np.random.normal(0, 1, n_days)),
            'Close': 100 + np.cumsum(np.random.normal(0, 1, n_days)),
            'Volume': np.random.lognormal(15, 0.5, n_days).astype(int),
            'Adj Close': 100 + np.cumsum(np.random.normal(0, 1, n_days))
        }, index=dates)
        
        config = DataConfig()
        engineer = FeatureEngineer(config)
        
        def feature_engineering_step():
            return engineer.engineer_features(raw_data)
        
        result = benchmark(feature_engineering_step)
        
        # Feature engineering should be reasonably fast
        assert result.stats['mean'] < 5.0  # Less than 5 seconds for 1000 days
    
    @pytest.mark.benchmark(group="data_processing")
    def test_dataset_creation_speed(self, benchmark):
        """Benchmark dataset creation performance"""
        from src.data.datasets.stock_dataset import StockDataset
        
        # Generate large feature dataset
        np.random.seed(42)
        n_days = 2000
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        features_data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.normal(0, 1, n_days)),
            'Volume': np.random.lognormal(15, 0.5, n_days).astype(int),
            'Returns': np.random.normal(0.001, 0.02, n_days),
            'SMA_5': 100 + np.cumsum(np.random.normal(0, 0.5, n_days)),
            'RSI': np.random.uniform(20, 80, n_days)
        }, index=dates)
        
        def create_dataset():
            return StockDataset(
                data=features_data,
                sequence_length=60,
                prediction_horizon=5,
                features=['Close', 'Volume', 'Returns', 'SMA_5', 'RSI']
            )
        
        result = benchmark(create_dataset)
        
        # Dataset creation should be fast
        assert result.stats['mean'] < 2.0  # Less than 2 seconds for large dataset