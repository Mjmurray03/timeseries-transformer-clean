# tests/integration/test_model_forward.py
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.timeseries_transformer import TimeSeriesTransformer

def test_model_forward_pass():
    """Test model initialization and forward pass."""
    
    print("[*] Testing Model Forward Pass...")
    
    # Test 1: Model Initialization
    print("\n1. Testing Model Initialization...")
    model_config = {
        'input_dim': 10,  # OHLCV + 5 indicators
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'max_seq_length': 60,
        'output_dim': 3  # [price, direction, volatility]
    }
    
    model = TimeSeriesTransformer(**model_config)
    assert model is not None, "[X] Model initialization failed"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model initialized: {trainable_params:,} trainable parameters")
    
    # Test 2: Forward Pass
    print("\n2. Testing Forward Pass...")
    batch_size = 32
    seq_length = 60
    input_dim = 10
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(dummy_input, return_attention=True)
    
    assert output is not None, "[X] No output from model"
    assert output.shape == (batch_size, 3), f"[X] Wrong output shape: {output.shape}"
    print(f"[OK] Forward pass successful: output shape {output.shape}")
    
    # Test 3: Attention Weights
    print("\n3. Testing Attention Extraction...")
    assert attention_weights is not None, "[X] No attention weights returned"
    assert len(attention_weights) == model_config['num_layers'], "[X] Wrong number of attention layers"
    
    for i, attn in enumerate(attention_weights):
        # PyTorch's MultiheadAttention returns averaged attention weights by default
        expected_shape = (batch_size, seq_length, seq_length)
        assert attn.shape == expected_shape, f"[X] Layer {i} wrong attention shape: got {attn.shape}, expected {expected_shape}"
    
    print(f"[OK] Attention extraction working: {len(attention_weights)} layers")
    
    # Test 4: GPU Compatibility (if available)
    print("\n4. Testing GPU Compatibility...")
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            output_gpu, _ = model(dummy_input, return_attention=False)
        
        assert output_gpu.is_cuda, "[X] Output not on GPU"
        print(f"[OK] GPU forward pass successful on {torch.cuda.get_device_name(0)}")
    else:
        print("[SKIP] No GPU available - skipping GPU test")
    
    print("\n[SUCCESS] Model forward pass test PASSED!")
    return True

if __name__ == "__main__":
    test_model_forward_pass()