import torch
import json
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_dummy_model():
    """Create dummy models and scalers for API testing."""
    
    print("=" * 60)
    print("CREATING DUMMY MODELS FOR TESTING")
    print("=" * 60)
    
    # Import model class
    from src.models.timeseries_transformer import TimeSeriesTransformer
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create scalers directory
    scalers_dir = Path("scalers")
    scalers_dir.mkdir(exist_ok=True)
    
    # List of tickers to create models for
    tickers = ['AAPL', 'MSFT', 'GOOG']
    
    for ticker in tickers:
        print(f"\nCreating model for {ticker}...")
        
        # Create model instance
        model = TimeSeriesTransformer(
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
        
        # Initialize weights with small random values
        for param in model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
        
        # Save model checkpoint
        model_path = models_dir / f"{ticker}_best.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': 10,
                'hidden_dim': 256,
                'num_heads': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'max_seq_length': 60,
                'output_dim': 3,
                'forecast_horizon': 5,
                'use_attention_pooling': True
            },
            'training_info': {
                'epochs_trained': 0,
                'best_loss': 0.001,
                'ticker': ticker
            }
        }
        
        torch.save(checkpoint, model_path)
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Model saved: {model_path.name} ({file_size_mb:.2f} MB)")
        
        # Create corresponding scaler
        scaler_data = {
            "mean": [100.0, 101.0, 99.0, 100.5, 1000000.0, 0.001, 0.02, 50.0, 0.0, 2.0],
            "std": [10.0, 10.0, 10.0, 10.0, 500000.0, 0.01, 0.05, 20.0, 1.0, 0.5],
            "min": [80.0, 81.0, 79.0, 80.5, 100000.0, -0.05, 0.0, 0.0, -2.0, 1.0],
            "max": [120.0, 121.0, 119.0, 120.5, 5000000.0, 0.05, 0.1, 100.0, 2.0, 3.0],
            "feature_names": [
                "open", "high", "low", "close", "volume",
                "returns", "volatility", "rsi", "macd", "bb_width"
            ],
            "scaler_type": "StandardScaler",
            "ticker": ticker
        }
        
        scaler_path = scalers_dir / f"scaler_{ticker}.json"
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        print(f"  [OK] Scaler saved: {scaler_path.name}")
    
    # Verify files were created
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    print("\nModels created:")
    for model_file in models_dir.glob("*.pt"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name}: {size_mb:.2f} MB")
    
    print("\nScalers created:")
    for scaler_file in scalers_dir.glob("*.json"):
        print(f"  - {scaler_file.name}")
    
    print("\n[SUCCESS] Dummy models and scalers created successfully!")
    return True

if __name__ == "__main__":
    success = create_dummy_model()
    sys.exit(0 if success else 1)