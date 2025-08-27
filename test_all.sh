#!/bin/bash
# test_all.sh

echo "RUNNING ALL INTEGRATION TESTS"
echo "================================"

# Test 1: Data Pipeline
echo -e "\nTEST 1: Data Pipeline"
doppler run -- python tests/integration/test_data_pipeline.py
if [ $? -ne 0 ]; then echo "ERROR: Data Pipeline Failed"; exit 1; fi

# Test 2: Model Forward
echo -e "\nTEST 2: Model Forward Pass"
doppler run -- python tests/integration/test_model_forward.py
if [ $? -ne 0 ]; then echo "ERROR: Model Forward Failed"; exit 1; fi

# Test 3: Training Components
echo -e "\nTEST 3: Training Components"
doppler run -- python tests/integration/test_training_components.py
if [ $? -ne 0 ]; then echo "ERROR: Training Components Failed"; exit 1; fi

# Test 4: W&B Integration
echo -e "\nTEST 4: W&B Integration"
doppler run -- python tests/integration/test_wandb_integration.py
if [ $? -ne 0 ]; then echo "ERROR: W&B Integration Failed"; exit 1; fi

# Test 5: End-to-End
echo -e "\nTEST 5: End-to-End Pipeline"
doppler run -- python tests/integration/test_end_to_end.py
if [ $? -ne 0 ]; then echo "ERROR: End-to-End Failed"; exit 1; fi

echo -e "\nSUCCESS: ALL INTEGRATION TESTS PASSED!"
echo "You're ready for Phase 5: Testing & Validation!"