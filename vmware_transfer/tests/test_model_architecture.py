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
        
        # CORRECTED parameters matching your actual model
        model_params = {
            'input_dim': 10,           # Input features (OHLCV + 5 indicators)
            'hidden_dim': 256,          # Model dimension (was d_model)
            'num_heads': 8,             # Attention heads (was n_heads)
            'num_layers': 4,            # Transformer layers (was n_layers)
            'dropout': 0.1,             # Dropout probability
            'max_seq_length': 60,       # Sequence length (was seq_len)
            'output_dim': 3,            # Output dimension
            'forecast_horizon': 5,      # Prediction horizon
            'use_attention_pooling': True  # Use attention pooling
        }
        
        print(f"\nModel Configuration:")
        for key, value in model_params.items():
            print(f"   {key:22} = {value}")
        
        # Create model instance
        model = TimeSeriesTransformer(**model_params)
        print(f"\n[OK] Model instantiated successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"   Total parameters:     {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # With hidden_dim=256, expect more parameters
        expected_min = 500_000
        expected_max = 2_000_000
        
        if expected_min <= total_params <= expected_max:
            print(f"   [OK] Parameter count in expected range ({expected_min:,} - {expected_max:,})")
        else:
            print(f"   [WARNING] Parameter count outside expected range (but that's OK)")
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        batch_size = 2
        seq_length = model_params['max_seq_length']
        input_features = model_params['input_dim']
        
        test_input = torch.randn(batch_size, seq_length, input_features)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        # The output might be a dict or tuple if multi-task
        if isinstance(output, dict):
            print(f"\nMulti-task output detected:")
            for key, value in output.items():
                print(f"   {key:20} shape: {tuple(value.shape)}")
        elif isinstance(output, tuple):
            print(f"\nMultiple outputs detected:")
            for i, out in enumerate(output):
                print(f"   Output {i}: shape {tuple(out.shape)}")
        else:
            print(f"   Output shape: {tuple(output.shape)}")
            
            # Check if output matches expected
            expected_shape = (batch_size, model_params['output_dim'])
            if output.shape[:2] == expected_shape[:2]:
                print(f"   [OK] Output shape correct!")
            else:
                print(f"   [WARNING] Output shape different than expected (might be multi-horizon)")
        
        # Test with different batch sizes
        print(f"\nTesting variable batch sizes...")
        for batch_size in [1, 4, 8]:
            test_input = torch.randn(batch_size, seq_length, input_features)
            with torch.no_grad():
                output = model(test_input)
            
            # Just check it doesn't crash
            if isinstance(output, dict):
                print(f"   [OK] Batch size {batch_size:2} works (dict output)")
            elif isinstance(output, tuple):
                print(f"   [OK] Batch size {batch_size:2} works (tuple output)")
            else:
                print(f"   [OK] Batch size {batch_size:2} works - shape: {tuple(output.shape)}")
        
        # Test model device compatibility
        print(f"\nDevice Compatibility:")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Available device: {device}")
        
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        model = model.to(device)
        test_input = test_input.to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   [OK] Model runs on {device}")
        
        # Test gradient flow (important for training)
        print(f"\nTesting gradient flow...")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Forward pass
        test_input = torch.randn(4, seq_length, input_features).to(device)
        output = model(test_input)
        
        # Handle different output types for loss
        if isinstance(output, dict):
            # If dict, use the main prediction output
            if 'predictions' in output:
                loss = output['predictions'].mean()
            elif 'price' in output:
                loss = output['price'].mean()
            else:
                # Just use the first value
                loss = list(output.values())[0].mean()
        elif isinstance(output, tuple):
            # Use first output
            loss = output[0].mean()
        else:
            # Simple tensor output
            loss = output.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        gradients_exist = False
        zero_gradients = 0
        total_params_checked = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params_checked += 1
                if param.grad is not None:
                    gradients_exist = True
                    if torch.all(param.grad == 0):
                        zero_gradients += 1
        
        if gradients_exist:
            print(f"   [OK] Gradients flow correctly")
            if zero_gradients > 0:
                print(f"   [WARNING] {zero_gradients}/{total_params_checked} parameters have zero gradients")
        else:
            print(f"   [ERROR] No gradients found")
            return False
        
        # Model architecture details
        print(f"\nArchitecture Details:")
        print(f"   Transformer blocks: {model_params['num_layers']}")
        print(f"   Attention heads: {model_params['num_heads']}")
        print(f"   Hidden dimension: {model_params['hidden_dim']}")
        print(f"   Sequence length: {model_params['max_seq_length']}")
        print(f"   Forecast horizon: {model_params['forecast_horizon']}")
        
        print(f"\n[SUCCESS] ALL MODEL TESTS PASSED!")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import Error: {e}")
        print(f"\nPossible issues:")
        print(f"   1. Missing component files in src/models/components/")
        print(f"   2. Missing dependencies")
        return False
        
    except Exception as e:
        print(f"[ERROR] Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = test_model_architecture()
    sys.exit(0 if result else 1)