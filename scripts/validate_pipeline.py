#!/usr/bin/env python3
"""
Pipeline Validation Script

Tests each component of the ML pipeline to ensure proper functioning.
Validates models, predictions, and data files for realistic values and proper formats.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime


def validate_predictions(file_path):
    """Validate prediction file format and values"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check structure
        if not isinstance(data, dict):
            return False, "Prediction file must be a JSON object"

        required_keys = ["ticker"]
        for key in required_keys:
            if key not in data:
                return False, f"Missing required key: {key}"

        # Check predictions are reasonable
        predictions_found = False

        # Handle different prediction formats
        if "prediction" in data and "predictions" in data["prediction"]:
            preds = data["prediction"]["predictions"]
            predictions_found = True
        elif "predictions" in data:
            preds = data["predictions"]
            predictions_found = True
        elif "rankings" in data:
            # Multi-ticker format
            preds = data["rankings"]
            predictions_found = True

        if predictions_found and isinstance(preds, list):
            for pred in preds:
                if isinstance(pred, dict):
                    # Check for realistic return values
                    return_keys = ["predicted_return_%", "adjusted_prediction", "avg_3day_return"]
                    for key in return_keys:
                        if key in pred:
                            val = pred[key]
                            if isinstance(val, (int, float)):
                                # Convert percentage to decimal if needed
                                if key == "predicted_return_%":
                                    val = val / 100 if abs(val) > 1 else val

                                # Check if return is realistic (max 20% daily)
                                if abs(val) > 0.20:
                                    return False, f"Unrealistic prediction: {val*100:.1f}% daily return"

        return True, "Valid prediction format"

    except json.JSONDecodeError:
        return False, "Invalid JSON format"
    except Exception as e:
        return False, str(e)


def validate_model(file_path):
    """Validate model file"""
    try:
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

        # Check for state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume direct state dict
            state_dict = checkpoint

        # Check for model configuration
        config_found = False
        if "model_config" in checkpoint and checkpoint["model_config"]:
            config = checkpoint["model_config"]
            config_found = True
        elif "training_args" in checkpoint:
            config = checkpoint["training_args"]
            config_found = True

        if config_found:
            # Validate architecture parameters
            if "hidden_dim" in config:
                hidden_dim = config["hidden_dim"]
                if hidden_dim not in [64, 128, 256, 512]:
                    return True, f"Unusual hidden_dim: {hidden_dim} (but acceptable)"

            if "input_dim" in config:
                input_dim = config["input_dim"]
                if input_dim != 10:
                    return True, f"Input_dim: {input_dim} (expected 10, may cause issues)"

        # Check model size
        param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        size_mb = file_path.stat().st_size / (1024 * 1024)

        return True, f"Valid model ({param_count:,} params, {size_mb:.1f}MB)"

    except Exception as e:
        return False, str(e)


def validate_market_data(file_path):
    """Validate market data file"""
    try:
        df = pd.read_csv(file_path)

        # Check required columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check date format
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            return False, "Invalid date format"

        # Check for reasonable price values
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                prices = df[col].dropna()
                if len(prices) > 0:
                    if prices.min() <= 0:
                        return False, f"Invalid {col} prices (negative or zero)"
                    if prices.max() > 10000:
                        return False, f"Unrealistic {col} prices (max: ${prices.max():.2f})"

        # Check data completeness
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.1:
            return False, f"Too much missing data ({missing_ratio:.1%})"

        return True, f"Valid market data ({len(df)} rows, {df['Date'].min()} to {df['Date'].max()})"

    except Exception as e:
        return False, str(e)


def validate_scaler(file_path):
    """Validate scaler file"""
    try:
        with open(file_path, 'r') as f:
            scaler_data = json.load(f)

        required_keys = ["mean", "std", "feature_names"]
        for key in required_keys:
            if key not in scaler_data:
                return False, f"Missing key: {key}"

        # Check dimensions match
        mean = scaler_data["mean"]
        std = scaler_data["std"]
        feature_names = scaler_data["feature_names"]

        if len(mean) != len(std) or len(mean) != len(feature_names):
            return False, "Dimension mismatch between mean, std, and feature_names"

        # Check for expected number of features (should be 10)
        if len(feature_names) != 10:
            return False, f"Expected 10 features, got {len(feature_names)}"

        # Check for reasonable scaler values
        mean_arr = np.array(mean)
        std_arr = np.array(std)

        if np.any(std_arr <= 0):
            return False, "Invalid standard deviation values (zero or negative)"

        return True, f"Valid scaler ({len(feature_names)} features)"

    except Exception as e:
        return False, str(e)


def main():
    """Main validation function"""
    print("=" * 60)
    print("PIPELINE VALIDATION REPORT")
    print("=" * 60)
    print(f"Validation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Validation results
    results = {
        "models": [],
        "predictions": [],
        "market_data": [],
        "scalers": []
    }

    # Check models
    print("📦 MODELS")
    print("-" * 20)
    model_dir = Path("models")
    if model_dir.exists():
        for model_file in model_dir.glob("*.pt"):
            valid, msg = validate_model(model_file)
            status = "✓" if valid else "✗"
            print(f"  {status} {model_file.name}: {msg}")
            results["models"].append({"file": str(model_file), "valid": valid, "message": msg})
    else:
        print("  ⚠ Models directory not found")

    print()

    # Check predictions
    print("🔮 PREDICTIONS")
    print("-" * 20)
    pred_dir = Path("predictions")
    if pred_dir.exists():
        for pred_file in pred_dir.glob("*.json"):
            valid, msg = validate_predictions(pred_file)
            status = "✓" if valid else "✗"
            print(f"  {status} {pred_file.name}: {msg}")
            results["predictions"].append({"file": str(pred_file), "valid": valid, "message": msg})
    else:
        print("  ⚠ Predictions directory not found")

    print()

    # Check market data
    print("📈 MARKET DATA")
    print("-" * 20)
    data_dir = Path("data/raw")
    if data_dir.exists():
        for data_file in data_dir.glob("*.csv"):
            valid, msg = validate_market_data(data_file)
            status = "✓" if valid else "✗"
            print(f"  {status} {data_file.name}: {msg}")
            results["market_data"].append({"file": str(data_file), "valid": valid, "message": msg})
    else:
        print("  ⚠ Market data directory not found")

    print()

    # Check scalers
    print("⚖️ SCALERS")
    print("-" * 20)
    scaler_dir = Path("scalers")
    if scaler_dir.exists():
        for scaler_file in scaler_dir.glob("*.json"):
            valid, msg = validate_scaler(scaler_file)
            status = "✓" if valid else "✗"
            print(f"  {status} {scaler_file.name}: {msg}")
            results["scalers"].append({"file": str(scaler_file), "valid": valid, "message": msg})
    else:
        print("  ⚠ Scalers directory not found")

    print()

    # Summary
    print("📊 SUMMARY")
    print("-" * 20)
    total_files = sum(len(results[key]) for key in results)
    valid_files = sum(len([r for r in results[key] if r["valid"]]) for key in results)

    print(f"Total files checked: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Success rate: {valid_files/total_files:.1%}" if total_files > 0 else "No files found")

    if valid_files == total_files and total_files > 0:
        print("\n🎉 All pipeline components are valid!")
    elif valid_files > 0:
        print(f"\n⚠️ {total_files - valid_files} issues found - check messages above")
    else:
        print("\n❌ No valid pipeline components found")

    print("\n" + "=" * 60)

    # Save validation report
    report_file = Path("results") / f"pipeline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": total_files,
                "valid_files": valid_files,
                "success_rate": valid_files/total_files if total_files > 0 else 0
            },
            "results": results
        }, f, indent=2)

    print(f"Detailed report saved: {report_file}")

    return 0 if valid_files == total_files and total_files > 0 else 1


if __name__ == "__main__":
    sys.exit(main())