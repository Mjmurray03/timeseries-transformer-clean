import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_architecture():
    """Test the TimeSeriesTransformer model architecture."""
    
    print("=" * 60)
    print("MODEL ARCHITECTURE TEST")
    print("=" * 60)
    
    try:
        # Import the model
        from src.models.timeseries_transformer import TimeSeriesTransformer
        print("[OK] Model import successful")
        
        # Standard parameters from your project
        model_params = {
            'input_dim': 8,      # Number of features
            'd_model': 128,      # Model dimension
            'n_heads': 8,        # Attention heads
            'n_layers': 4,       # Transformer layers
            'd_ff': 512,         # Feed-forward dimension
            'seq_len': 60,       # Sequence length
            'output_dim': 3      # 3-day predictions
        }
        
        print(f"\nModel Configuration:")
        for key, value in model_params.items():
            print(f"   {key:15} = {value}")
        
        # Create model instance
        model = TimeSeriesTransformer(**model_params)
        print(f"\n[OK] Model instantiated successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"   Total parameters:     {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Expected parameters:  464,571")
        
        if abs(total_params - 464571) < 1000:  # Allow small variance
            print(f"   [OK] Parameter count matches expected!")
        else:
            print(f"   [WARNING] Parameter count differs from expected")
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        batch_size = 2
        test_input = torch.randn(batch_size, model_params['seq_len'], model_params['input_dim'])
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        expected_shape = (batch_size, model_params['output_dim'])
        actual_shape = tuple(output.shape)
        
        print(f"   Input shape:    {tuple(test_input.shape)}")
        print(f"   Output shape:   {actual_shape}")
        print(f"   Expected shape: {expected_shape}")
        
        if actual_shape == expected_shape:
            print(f"   [OK] Output shape correct!")
        else:
            print(f"   [ERROR] Output shape mismatch!")
            return False
        
        # Test model device compatibility
        print(f"\nDevice Compatibility:")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Available device: {device}")
        
        model = model.to(device)
        test_input = test_input.to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   [OK] Model runs on {device}")
        
        print(f"\n[SUCCESS] ALL MODEL TESTS PASSED!")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import Error: {e}")
        print(f"   Make sure you're in the project root and virtual environment is activated")
        return False
    except Exception as e:
        print(f"[ERROR] Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_model_architecture()
    sys.exit(0 if result else 1)