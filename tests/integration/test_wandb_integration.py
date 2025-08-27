# tests/integration/test_wandb_integration.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import wandb
import numpy as np
import os

def test_wandb_integration():
    """Test Weights & Biases integration with Doppler."""
    
    print("* Testing W&B Integration...")
    
    # Test 1: W&B Initialization
    print("\n1. Testing W&B Initialization...")
    
    # Check if API key is available from Doppler
    api_key = os.getenv('WANDB_API_KEY')
    assert api_key is not None, "ERROR: WANDB_API_KEY not found in environment"
    print("SUCCESS: W&B API key found from Doppler")
    
    # Initialize W&B run
    run = wandb.init(
        project='timeseries-transformer-test',
        name='integration-test',
        config={
            'test': True,
            'learning_rate': 0.001,
            'batch_size': 32,
            'model': 'TimeSeriesTransformer'
        },
        mode='offline'  # Use offline mode for testing
    )
    
    assert run is not None, "ERROR: W&B initialization failed"
    print(f"SUCCESS: W&B run initialized: {run.name}")
    
    # Test 2: Log Metrics
    print("\n2. Testing Metric Logging...")
    
    for i in range(5):
        metrics = {
            'train/loss': np.random.random(),
            'train/accuracy': np.random.random(),
            'val/loss': np.random.random(),
            'val/accuracy': np.random.random(),
            'learning_rate': 0.001 * (0.9 ** i)
        }
        
        wandb.log(metrics, step=i)
    
    print("SUCCESS: Metrics logged successfully")
    
    # Test 3: Log Model Architecture
    print("\n3. Testing Model Summary Logging...")
    
    model_summary = {
        'architecture': 'Transformer',
        'parameters': 1500000,
        'layers': 4,
        'heads': 8,
        'hidden_dim': 256
    }
    
    wandb.config.update(model_summary)
    print("SUCCESS: Model summary logged")
    
    # Test 4: Log Table Data
    print("\n4. Testing Table Logging...")
    
    table = wandb.Table(columns=['epoch', 'train_loss', 'val_loss', 'best'])
    table.add_data(1, 0.5, 0.6, False)
    table.add_data(2, 0.4, 0.5, False)
    table.add_data(3, 0.3, 0.4, True)
    
    wandb.log({'performance_table': table})
    print("SUCCESS: Table data logged")
    
    # Finish run
    wandb.finish()
    
    print("\nSUCCESS: W&B integration test PASSED!")
    print("NOTE: Run was in offline mode. Use mode='online' for actual training.")
    return True

if __name__ == "__main__":
    test_wandb_integration()