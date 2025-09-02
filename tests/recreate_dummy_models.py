import torch
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def recreate_dummy_models():
    """Recreate dummy models with correct checkpoint structure."""
    
    print("=" * 60)
    print("RECREATING DUMMY MODELS")
    print("=" * 60)
    
    # Import model class
    from src.models.timeseries_transformer import TimeSeriesTransformer
    
    models_dir = Path("models")
    scalers_dir = Path("scalers")
    
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
        
        # Save JUST the state dict (not nested)
        model_path = models_dir / f"{ticker}_best.pt"
        
        # Save directly as state_dict (no nesting)
        torch.save(model.state_dict(), model_path)
        
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] Model saved: {model_path.name} ({file_size_mb:.2f} MB)")
    
    print("\n[SUCCESS] Models recreated with correct structure")
    return True

if __name__ == "__main__":
    recreate_dummy_models()