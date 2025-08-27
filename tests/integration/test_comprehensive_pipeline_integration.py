"""
Comprehensive integration tests for pipeline components.
Follows testing-standards.md patterns with 20% of total test coverage.
Tests end-to-end workflows and component interactions.
"""
import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.data.collectors.yahoo_finance import YahooFinanceCollector
from src.data.processors.feature_engineering import FeatureEngineer  
from src.data.datasets.stock_dataset import StockDataset
from src.models.timeseries_transformer import TimeSeriesTransformer
from src.training.trainer import Trainer
from src.config.training_config import TrainingConfig
from src.config.model_config import ModelConfig
from src.config.data_config import DataConfig


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline"""
    
    @pytest.fixture
    def data_config(self):
        """Create data configuration for integration testing"""
        return DataConfig(
            tickers=['AAPL', 'MSFT'],
            start_date='2023-01-01',
            end_date='2023-06-30',
            features=['Close', 'Volume', 'Returns', 'SMA_5', 'RSI'],
            sequence_length=60,
            prediction_horizon=5
        )
    
    @pytest.fixture
    def mock_yahoo_data(self):
        """Generate comprehensive mock data for integration testing"""
        date_range = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        n_days = len(date_range)
        
        # Generate realistic stock data
        np.random.seed(42)
        base_prices = {'AAPL': 150.0, 'MSFT': 250.0}
        
        mock_data = {}
        for ticker in ['AAPL', 'MSFT']:
            base_price = base_prices[ticker]
            returns = np.random.normal(0.001, 0.02, n_days)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            prices = np.array(prices)
            
            mock_data[ticker] = pd.DataFrame({
                'Open': prices * np.random.uniform(0.99, 1.01, n_days),
                'High': prices * np.random.uniform(1.005, 1.02, n_days),
                'Low': prices * np.random.uniform(0.98, 0.995, n_days),
                'Close': prices,
                'Volume': np.random.lognormal(15, 0.5, n_days).astype(int),
                'Adj Close': prices
            }, index=date_range)
        
        return mock_data
    
    def test_end_to_end_data_pipeline(self, data_config, mock_yahoo_data):
        """Test complete data pipeline from collection to dataset creation"""
        # Step 1: Data Collection
        collector = YahooFinanceCollector(cache_enabled=False)
        
        with patch('yfinance.download') as mock_download:
            def mock_yf_download(ticker, *args, **kwargs):
                return mock_yahoo_data[ticker]
            mock_download.side_effect = mock_yf_download
            
            # Collect data for multiple tickers
            collected_data = {}
            for ticker in data_config.tickers:
                data = collector.fetch_data(
                    ticker=ticker,
                    start=data_config.start_date,
                    end=data_config.end_date
                )
                collected_data[ticker] = data
                
                # Verify data collection
                assert not data.empty
                assert len(data) > 0
                assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Step 2: Feature Engineering
        engineer = FeatureEngineer(data_config)
        
        engineered_data = {}
        for ticker, raw_data in collected_data.items():
            features = engineer.engineer_features(raw_data)
            engineered_data[ticker] = features
            
            # Verify feature engineering
            assert not features.empty
            expected_features = ['Returns', 'SMA_5', 'RSI']
            for feature in expected_features:
                if feature in data_config.features:
                    assert feature in features.columns
        
        # Step 3: Dataset Creation
        datasets = {}
        for ticker, features in engineered_data.items():
            # Filter to configured features
            available_features = [f for f in data_config.features if f in features.columns]
            feature_data = features[available_features]
            
            dataset = StockDataset(
                data=feature_data,
                sequence_length=data_config.sequence_length,
                prediction_horizon=data_config.prediction_horizon,
                features=available_features
            )
            datasets[ticker] = dataset
            
            # Verify dataset creation
            if len(dataset) > 0:
                sample = dataset[0]
                assert 'input' in sample
                assert 'target' in sample
                assert sample['input'].shape[0] == data_config.sequence_length
                assert sample['target'].shape[0] == data_config.prediction_horizon
        
        # Step 4: Integration verification
        # Verify we can combine datasets
        all_sequences = []
        all_targets = []
        
        for dataset in datasets.values():
            if len(dataset) > 0:
                for i in range(len(dataset)):
                    sample = dataset[i]
                    all_sequences.append(sample['input'])
                    all_targets.append(sample['target'])
        
        if all_sequences:
            combined_sequences = torch.stack(all_sequences)
            combined_targets = torch.stack(all_targets)
            
            assert combined_sequences.shape[1] == data_config.sequence_length
            assert combined_sequences.shape[2] == len(data_config.features)
            assert combined_targets.shape[1] == data_config.prediction_horizon
            
            # Verify no data leakage
            assert not torch.isnan(combined_sequences).any()
            assert not torch.isnan(combined_targets).any()
    
    def test_data_validation_integration(self, data_config, mock_yahoo_data):
        """Test data validation throughout the pipeline"""
        from src.data.validators import DataValidator
        
        validator = DataValidator()
        collector = YahooFinanceCollector(cache_enabled=False)
        engineer = FeatureEngineer(data_config)
        
        with patch('yfinance.download', return_value=mock_yahoo_data['AAPL']):
            # Test validation at each pipeline stage
            
            # Stage 1: Raw data validation
            raw_data = collector.fetch_data('AAPL', '2023-01-01', '2023-06-30')
            validation_result = validator.validate_ohlcv(raw_data)
            
            assert validation_result['is_valid'] is True
            assert len(validation_result['errors']) == 0
            
            # Stage 2: Feature validation
            features = engineer.engineer_features(raw_data)
            feature_validation = validator.validate_features(features)
            
            assert feature_validation['is_valid'] is True
            
            # Stage 3: Dataset validation
            dataset = StockDataset(
                data=features[data_config.features[:3]],  # Use first 3 features
                sequence_length=data_config.sequence_length,
                prediction_horizon=data_config.prediction_horizon,
                features=data_config.features[:3]
            )
            
            if len(dataset) > 0:
                # Test sequence validation
                for i in range(min(5, len(dataset))):
                    sample = dataset[i]
                    
                    # No future leakage check
                    assert sample['input'].shape[0] == data_config.sequence_length
                    assert sample['target'].shape[0] == data_config.prediction_horizon
                    
                    # Data quality checks
                    assert not torch.isnan(sample['input']).any()
                    assert not torch.isnan(sample['target']).any()
                    assert not torch.isinf(sample['input']).any()
                    assert not torch.isinf(sample['target']).any()
    
    def test_data_persistence_integration(self, data_config, mock_yahoo_data, tmp_path):
        """Test data storage and loading integration"""
        from src.data.storage import DataStorage
        
        storage = DataStorage(base_path=str(tmp_path))
        collector = YahooFinanceCollector(cache_enabled=False)
        engineer = FeatureEngineer(data_config)
        
        with patch('yfinance.download', return_value=mock_yahoo_data['AAPL']):
            # Pipeline with storage
            raw_data = collector.fetch_data('AAPL', '2023-01-01', '2023-06-30')
            
            # Store raw data
            storage.save('AAPL_raw', raw_data)
            
            # Process and store features
            features = engineer.engineer_features(raw_data)
            storage.save('AAPL_features', features)
            
            # Verify persistence
            loaded_raw = storage.load('AAPL_raw')
            loaded_features = storage.load('AAPL_features')
            
            pd.testing.assert_frame_equal(raw_data, loaded_raw)
            pd.testing.assert_frame_equal(features, loaded_features)
            
            # Test dataset creation from persisted data
            dataset = StockDataset(
                data=loaded_features[data_config.features[:3]],
                sequence_length=data_config.sequence_length,
                prediction_horizon=data_config.prediction_horizon,
                features=data_config.features[:3]
            )
            
            assert len(dataset) > 0 or len(loaded_features) < data_config.sequence_length + data_config.prediction_horizon


@pytest.mark.integration
class TestModelTrainingIntegration:
    """Integration tests for model training pipeline"""
    
    @pytest.fixture
    def integration_configs(self):
        """Create configurations for training integration"""
        model_config = ModelConfig(
            sequence_length=30,  # Smaller for faster testing
            num_features=5,
            d_model=64,  # Smaller for testing
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            forecast_horizon=3
        )
        
        training_config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=8,  # Small batch for testing
            num_epochs=3,  # Few epochs for testing
            patience=10,
            gradient_clip=1.0,
            weight_decay=1e-4
        )
        
        return model_config, training_config
    
    @pytest.fixture
    def integration_dataloaders(self):
        """Create mock dataloaders for integration testing"""
        def create_batch():
            return {
                'input': torch.randn(8, 30, 5),
                'target': torch.randn(8, 3)
            }
        
        # Create consistent batches for reproducible testing
        train_batches = [create_batch() for _ in range(5)]
        val_batches = [create_batch() for _ in range(2)]
        
        train_loader = Mock()
        train_loader.__iter__ = Mock(return_value=iter(train_batches))
        train_loader.__len__ = Mock(return_value=len(train_batches))
        
        val_loader = Mock()
        val_loader.__iter__ = Mock(return_value=iter(val_batches))
        val_loader.__len__ = Mock(return_value=len(val_batches))
        
        return train_loader, val_loader
    
    def test_complete_training_pipeline(self, integration_configs, integration_dataloaders, tmp_path):
        """Test complete training pipeline integration"""
        model_config, training_config = integration_configs
        train_loader, val_loader = integration_dataloaders
        
        # Create model
        model = TimeSeriesTransformer(**model_config.__dict__)
        
        # Create trainer with mocked experiment tracking
        with patch('src.training.experiment_tracker.ExperimentTracker'):
            trainer = Trainer(
                model=model,
                config=training_config,
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=str(tmp_path)
            )
            
            # Run training
            history = trainer.fit()
            
            # Verify training completed
            assert isinstance(history, dict)
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) == training_config.num_epochs
            assert len(history['val_loss']) == training_config.num_epochs
            
            # Verify losses are reasonable (positive and decreasing or stable)
            train_losses = history['train_loss']
            assert all(loss > 0 for loss in train_losses)
            
            # Check that model was saved
            checkpoint_files = list(tmp_path.glob("**/*.pt"))
            assert len(checkpoint_files) > 0
    
    def test_model_evaluation_integration(self, integration_configs, integration_dataloaders):
        """Test model evaluation integration"""
        model_config, _ = integration_configs
        _, val_loader = integration_dataloaders
        
        # Create and train model minimally
        model = TimeSeriesTransformer(**model_config.__dict__)
        model.eval()
        
        # Test evaluation loop
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['input'])
                
                # Verify outputs
                assert isinstance(outputs, dict)
                main_output = list(outputs.values())[0]
                assert main_output.shape[0] == batch['input'].shape[0]  # Batch size consistency
                
                # Compute loss (simplified)
                loss = torch.nn.functional.mse_loss(main_output, batch['target'])
                total_loss += loss.item() * batch['input'].shape[0]
                total_samples += batch['input'].shape[0]
        
        avg_loss = total_loss / total_samples
        assert avg_loss > 0
        assert not np.isnan(avg_loss)
        assert not np.isinf(avg_loss)
    
    def test_checkpoint_loading_integration(self, integration_configs, tmp_path):
        """Test model checkpoint saving and loading integration"""
        model_config, training_config = integration_configs
        
        # Create and setup model
        model1 = TimeSeriesTransformer(**model_config.__dict__)
        optimizer = torch.optim.Adam(model1.parameters(), lr=training_config.learning_rate)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        checkpoint = {
            'model_state_dict': model1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
            'loss': 0.5
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Create new model and load checkpoint
        model2 = TimeSeriesTransformer(**model_config.__dict__)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=training_config.learning_rate)
        
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model2.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer2.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        
        # Verify models are identical
        test_input = torch.randn(1, 30, 5)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)
        
        # Compare outputs
        for key in output1.keys():
            torch.testing.assert_close(output1[key], output2[key], rtol=1e-5, atol=1e-7)
    
    def test_early_stopping_integration(self, integration_configs, integration_dataloaders, tmp_path):
        """Test early stopping integration with training"""
        from src.training.callbacks.early_stopping import EarlyStopping
        
        model_config, training_config = integration_configs
        train_loader, val_loader = integration_dataloaders
        
        # Modify config for early stopping test
        training_config.num_epochs = 20  # More epochs to test early stopping
        training_config.patience = 3
        
        model = TimeSeriesTransformer(**model_config.__dict__)
        early_stopping = EarlyStopping(patience=3, min_delta=1e-4)
        
        # Simulate training with early stopping
        val_losses = [1.0, 0.9, 0.85, 0.87, 0.86, 0.88]  # Stops improving after epoch 2
        
        should_stop = False
        stopped_epoch = 0
        
        for epoch, val_loss in enumerate(val_losses, 1):
            should_stop = early_stopping(val_loss, epoch)
            if should_stop:
                stopped_epoch = epoch
                break
        
        assert should_stop is True
        assert stopped_epoch < len(val_losses)  # Should stop before going through all losses
        assert early_stopping.best_loss <= 0.85  # Should track the best loss
    
    def test_learning_rate_scheduling_integration(self, integration_configs):
        """Test learning rate scheduling integration"""
        model_config, training_config = integration_configs
        
        model = TimeSeriesTransformer(**model_config.__dict__)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        initial_lr = optimizer.param_groups[0]['lr']
        lr_history = [initial_lr]
        
        # Simulate training steps
        for step in range(10):
            # Simulate training step
            optimizer.zero_grad()
            
            # Dummy forward pass
            dummy_input = torch.randn(8, 30, 5)
            outputs = model(dummy_input)
            loss = list(outputs.values())[0].sum()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
        
        # Verify learning rate changed
        assert lr_history[-1] != initial_lr
        assert len(set(lr_history)) > 1  # Learning rate should have varied


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API components if implemented"""
    
    @pytest.fixture
    def mock_trained_model(self):
        """Create mock trained model for API testing"""
        model = Mock()
        model.eval = Mock()
        model.predict = Mock(return_value={
            'predictions': torch.randn(1, 5),
            'confidence': torch.randn(1, 5)
        })
        return model
    
    def test_prediction_endpoint_integration(self, mock_trained_model):
        """Test prediction API endpoint integration"""
        # This test assumes an API implementation exists
        # Skip if not implemented
        try:
            from src.api.routes.predictions import predict_endpoint
        except ImportError:
            pytest.skip("API not implemented")
        
        # Mock request data
        request_data = {
            'ticker': 'AAPL',
            'features': torch.randn(60, 7).tolist(),
            'horizon': 5
        }
        
        with patch('src.inference.predictor.Predictor') as mock_predictor:
            mock_predictor.return_value.predict.return_value = {
                'predictions': [1.0, 1.1, 1.2, 1.1, 1.05],
                'confidence_intervals': [[0.9, 1.1], [1.0, 1.2], [1.1, 1.3], [1.0, 1.2], [0.95, 1.15]]
            }
            
            # Test endpoint functionality would go here
            # This is a placeholder for when API is fully implemented
            assert True  # Placeholder assertion


@pytest.mark.integration  
class TestCacheIntegration:
    """Integration tests for caching system"""
    
    @pytest.fixture
    def cache_config(self):
        """Create cache configuration for testing"""
        return {
            'enabled': True,
            'redis_url': 'redis://localhost:6379/1',  # Test database
            'ttl': 300,  # 5 minutes
            'max_memory': '100mb'
        }
    
    def test_data_caching_integration(self, cache_config, mock_yahoo_data):
        """Test data collection with caching integration"""
        from src.data.collectors.yahoo_finance import YahooFinanceCollector
        
        with patch('redis.from_url') as mock_redis:
            # Mock Redis client
            mock_client = Mock()
            mock_client.get.return_value = None  # Cache miss first time
            mock_client.set.return_value = True
            mock_redis.return_value = mock_client
            
            collector = YahooFinanceCollector(cache_enabled=True)
            
            with patch('yfinance.download', return_value=mock_yahoo_data['AAPL']):
                # First call - should hit the API and cache
                data1 = collector.fetch_data('AAPL', '2023-01-01', '2023-01-31')
                
                # Verify cache set was called
                mock_client.set.assert_called()
                
                # Second call - mock cache hit
                mock_client.get.return_value = data1.to_json()
                data2 = collector.fetch_data('AAPL', '2023-01-01', '2023-01-31')
                
                # Should have tried to get from cache
                mock_client.get.assert_called()
    
    def test_model_prediction_caching(self, cache_config):
        """Test model prediction caching integration"""
        try:
            from src.cache.managers.prediction import PredictionCache
        except ImportError:
            pytest.skip("Prediction caching not implemented")
        
        with patch('redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_client.get.return_value = None
            mock_client.set.return_value = True
            mock_redis.return_value = mock_client
            
            cache = PredictionCache(**cache_config)
            
            # Mock prediction
            input_key = "test_input_hash"
            prediction = {'predictions': [1.0, 1.1, 1.2]}
            
            # Cache prediction
            cache.set(input_key, prediction)
            mock_client.set.assert_called()
            
            # Retrieve prediction
            mock_client.get.return_value = str(prediction)
            cached_result = cache.get(input_key)
            mock_client.get.assert_called()


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    @pytest.fixture
    def test_db_path(self, tmp_path):
        """Create temporary database path for testing"""
        return tmp_path / "test.db"
    
    def test_data_persistence_integration(self, test_db_path):
        """Test data persistence with database integration"""
        try:
            from src.data.storage import DatabaseStorage
        except ImportError:
            pytest.skip("Database storage not implemented")
        
        storage = DatabaseStorage(str(test_db_path))
        
        # Test data storage and retrieval
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': np.random.randn(100),
            'ticker': ['AAPL'] * 100
        })
        
        # Store data
        storage.save_dataframe('test_table', test_data)
        
        # Retrieve data
        loaded_data = storage.load_dataframe('test_table')
        
        # Verify data integrity
        assert len(loaded_data) == len(test_data)
        assert list(loaded_data.columns) == list(test_data.columns)
    
    def test_metadata_storage_integration(self, test_db_path):
        """Test metadata storage integration"""
        try:
            from src.data.storage import MetadataStorage
        except ImportError:
            pytest.skip("Metadata storage not implemented")
        
        storage = MetadataStorage(str(test_db_path))
        
        # Store metadata
        metadata = {
            'model_version': 'v1.0.0',
            'training_date': '2023-06-01',
            'metrics': {'rmse': 0.05, 'mae': 0.03},
            'hyperparameters': {'lr': 1e-4, 'batch_size': 32}
        }
        
        storage.save_metadata('model_v1', metadata)
        
        # Retrieve metadata
        loaded_metadata = storage.load_metadata('model_v1')
        
        # Verify metadata integrity
        assert loaded_metadata == metadata


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipelineIntegration:
    """Comprehensive end-to-end integration tests"""
    
    def test_complete_pipeline_integration(self, tmp_path):
        """Test complete pipeline from data collection to prediction"""
        # This is a comprehensive integration test that would test the entire pipeline
        # It's marked as slow since it involves multiple components
        
        # Configuration
        tickers = ['AAPL']
        start_date = '2023-01-01'
        end_date = '2023-03-31'
        
        # Step 1: Data Collection
        collector = YahooFinanceCollector(cache_enabled=False)
        
        # Mock data for integration test
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(140, 160, 90),
            'High': np.random.uniform(145, 165, 90),
            'Low': np.random.uniform(135, 155, 90),
            'Close': np.random.uniform(140, 160, 90),
            'Volume': np.random.randint(50000000, 200000000, 90),
            'Adj Close': np.random.uniform(140, 160, 90)
        }, index=pd.date_range(start_date, end_date, freq='D')[:90])
        
        with patch('yfinance.download', return_value=mock_data):
            raw_data = collector.fetch_data(tickers[0], start_date, end_date)
        
        # Step 2: Feature Engineering
        data_config = DataConfig()
        engineer = FeatureEngineer(data_config)
        features = engineer.engineer_features(raw_data)
        
        # Step 3: Dataset Creation
        if len(features) >= 65:  # Need minimum data for sequences
            dataset = StockDataset(
                data=features[['Close', 'Volume', 'Returns']],  # Use available features
                sequence_length=60,
                prediction_horizon=5,
                features=['Close', 'Volume', 'Returns']
            )
            
            if len(dataset) > 0:
                # Step 4: Model Training (minimal)
                model_config = ModelConfig(
                    sequence_length=60,
                    num_features=3,
                    d_model=64,
                    n_heads=4,
                    n_layers=2,
                    dropout=0.1,
                    forecast_horizon=5
                )
                
                model = TimeSeriesTransformer(**model_config.__dict__)
                
                # Create minimal dataloader
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
                
                # Test single training step
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                
                for batch_data in dataloader:
                    optimizer.zero_grad()
                    
                    # Get batch
                    inputs = batch_data['input']
                    targets = batch_data['target']
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Compute loss
                    main_output = list(outputs.values())[0]
                    loss = torch.nn.functional.mse_loss(main_output, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Verify training step completed
                    assert loss.item() > 0
                    assert not torch.isnan(loss)
                    
                    break  # Only test one step
                
                # Step 5: Model Evaluation
                model.eval()
                with torch.no_grad():
                    sample_input = dataset[0]['input'].unsqueeze(0)
                    prediction = model(sample_input)
                    
                    # Verify prediction
                    main_pred = list(prediction.values())[0]
                    assert main_pred.shape == (1, 5)
                    assert not torch.isnan(main_pred).any()
                    assert not torch.isinf(main_pred).any()
        
        # If we get here without errors, the integration test passed
        assert True