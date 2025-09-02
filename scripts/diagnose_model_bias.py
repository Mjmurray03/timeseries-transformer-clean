"""
Diagnostic script to analyze why model is producing only bullish predictions
"""

import json

# Import model
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.append(str(Path(__file__).parent.parent))
from src.models.timeseries_transformer import TimeSeriesTransformer


def load_model_and_analyze():
    """Load the trained model and analyze its predictions"""

    # Load the checkpoint
    checkpoint = torch.load("model_extended_best.pt", map_location="cpu")

    print("=" * 60)
    print("MODEL CHECKPOINT ANALYSIS")
    print("=" * 60)

    # Check training history
    if "history" in checkpoint:
        history = checkpoint["history"]
        print(f"\nTraining stopped at epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")

        # Analyze loss progression
        train_losses = history["train_loss"]
        val_losses = history["val_loss"]

        print(f"\nLoss Statistics:")
        print(f"  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Final val loss: {val_losses[-1]:.4f}")
        print(f"  Min val loss: {min(val_losses):.4f}")
        print(f"  Overfitting ratio: {train_losses[-1]/val_losses[-1]:.2f}")

    # Load model state
    model_state = checkpoint["model_state_dict"]

    # Analyze weight statistics
    print("\n" + "=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)

    for name, param in model_state.items():
        if "weight" in name:
            weights = param.cpu().numpy()
            print(f"\n{name}:")
            print(f"  Mean: {weights.mean():.6f}")
            print(f"  Std: {weights.std():.6f}")
            print(f"  Min: {weights.min():.6f}")
            print(f"  Max: {weights.max():.6f}")
            print(f"  % near zero (<0.001): {(np.abs(weights) < 0.001).mean()*100:.1f}%")

    return checkpoint


def analyze_prediction_patterns():
    """Analyze patterns in saved predictions if available"""

    # Try to load any saved predictions
    predictions_path = Path("predictions")
    if predictions_path.exists():
        pred_files = list(predictions_path.glob("*.json"))

        if pred_files:
            print("\n" + "=" * 60)
            print("SAVED PREDICTIONS ANALYSIS")
            print("=" * 60)

            for pred_file in pred_files[-3:]:  # Last 3 prediction files
                with open(pred_file, "r") as f:
                    data = json.load(f)

                print(f"\n{pred_file.name}:")

                # Analyze predictions
                all_predictions = []
                for stock, info in data.items():
                    if isinstance(info, dict) and "predictions" in info:
                        preds = info["predictions"]
                        for day, pred_info in preds.items():
                            if isinstance(pred_info, dict) and "change_percent" in pred_info:
                                all_predictions.append(pred_info["change_percent"])

                if all_predictions:
                    all_predictions = np.array(all_predictions)
                    print(f"  Total predictions: {len(all_predictions)}")
                    print(f"  Mean: {all_predictions.mean():.2f}%")
                    print(f"  Std: {all_predictions.std():.2f}%")
                    print(f"  Min: {all_predictions.min():.2f}%")
                    print(f"  Max: {all_predictions.max():.2f}%")
                    print(f"  % Bullish: {(all_predictions > 0).mean()*100:.1f}%")
                    print(f"  % Bearish: {(all_predictions < 0).mean()*100:.1f}%")


def test_model_on_synthetic_data():
    """Test model behavior on synthetic bearish/bullish data"""

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA TEST")
    print("=" * 60)

    # Load model
    checkpoint = torch.load("model_extended_best.pt", map_location="cpu")

    # Recreate model (using your architecture)
    model = TimeSeriesTransformer(
        input_dim=21,  # From training
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        forecast_horizon=3,
        output_dim=3,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create synthetic test cases
    test_cases = {
        "Strong Uptrend": np.linspace(0, 1, 60 * 21).reshape(60, 21),
        "Strong Downtrend": np.linspace(1, 0, 60 * 21).reshape(60, 21),
        "Flat/Sideways": np.ones((60, 21)) * 0.5,
        "Volatile": np.random.randn(60, 21) * 0.5,
        "Crash Pattern": np.concatenate([np.ones((40, 21)), np.zeros((20, 21))], axis=0),
    }

    print("\nModel response to different patterns:")
    print("-" * 40)

    results = {}
    for pattern_name, pattern_data in test_cases.items():
        # Convert to tensor
        x = torch.tensor(pattern_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(x).squeeze().numpy()

        results[pattern_name] = pred

        print(f"\n{pattern_name}:")
        print(f"  Predictions: {pred}")
        print(f"  Mean: {pred.mean():.4f}")
        print(f"  All positive?: {all(p > 0 for p in pred)}")

    return results


def analyze_training_data_distribution():
    """Check if training data itself was biased"""

    print("\n" + "=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)

    # Download recent data to check market conditions
    import yfinance as yf

    tickers = ["AAPL", "MSFT", "NVDA"]

    for ticker in tickers:
        df = yf.download(ticker, period="5y", progress=False)

        # Calculate 3-day returns
        returns_3d = (df["Close"].shift(-3) - df["Close"]) / df["Close"]
        returns_3d = returns_3d.dropna()

        print(f"\n{ticker} - Actual 3-day returns (last 5 years):")
        print(f"  Mean: {returns_3d.mean()*100:.2f}%")
        print(f"  Std: {returns_3d.std()*100:.2f}%")
        print(f"  % Positive: {(returns_3d > 0).mean()*100:.1f}%")
        print(f"  % Negative: {(returns_3d < 0).mean()*100:.1f}%")

        # Check last year specifically
        last_year = returns_3d[-252:]  # ~252 trading days per year
        print(f"\n  Last year only:")
        print(f"    Mean: {last_year.mean()*100:.2f}%")
        print(f"    % Positive: {(last_year > 0).mean()*100:.1f}%")


def create_diagnostic_plots():
    """Create comprehensive diagnostic plots"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Load checkpoint for history
    checkpoint = torch.load("model_extended_best.pt", map_location="cpu")

    if "history" in checkpoint:
        history = checkpoint["history"]

        # Plot 1: Loss curves
        axes[0, 0].plot(history["train_loss"], label="Train", linewidth=2)
        axes[0, 0].plot(history["val_loss"], label="Validation", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training History")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Learning rate
        axes[0, 1].plot(history["lr"], linewidth=2, color="orange")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Learning Rate")
        axes[0, 1].set_title("Learning Rate Schedule")
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Weight distribution
    model_state = checkpoint["model_state_dict"]
    all_weights = []
    for name, param in model_state.items():
        if "weight" in name:
            all_weights.extend(param.cpu().numpy().flatten())

    axes[0, 2].hist(all_weights, bins=50, alpha=0.7, color="green")
    axes[0, 2].axvline(x=0, color="red", linestyle="--", label="Zero")
    axes[0, 2].set_xlabel("Weight Value")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Weight Distribution")
    axes[0, 2].legend()

    # Plot 4: Test synthetic patterns
    results = test_model_on_synthetic_data()

    patterns = list(results.keys())
    means = [results[p].mean() for p in patterns]

    axes[1, 0].bar(range(len(patterns)), means)
    axes[1, 0].set_xticks(range(len(patterns)))
    axes[1, 0].set_xticklabels(patterns, rotation=45, ha="right")
    axes[1, 0].axhline(y=0, color="red", linestyle="--")
    axes[1, 0].set_ylabel("Mean Prediction")
    axes[1, 0].set_title("Model Response to Synthetic Patterns")

    # Plot 5: Actual market data distribution
    import yfinance as yf

    df = yf.download("SPY", period="5y", progress=False)
    returns_3d = (df["Close"].shift(-3) - df["Close"]) / df["Close"]
    returns_3d = returns_3d.dropna() * 100  # Convert to percentage

    axes[1, 1].hist(returns_3d, bins=50, alpha=0.7, color="blue")
    axes[1, 1].axvline(x=0, color="red", linestyle="--", label="Zero")
    axes[1, 1].set_xlabel("3-Day Return (%)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Actual SPY 3-Day Returns (5 Years)")
    axes[1, 1].legend()

    # Plot 6: Prediction bias over time
    axes[1, 2].text(
        0.5,
        0.5,
        "Model Bias Analysis\n\nCurrent Issue:\nAll predictions positive\n\nLikely Causes:\n1. Loss function imbalance\n2. Recent bull market bias\n3. Penalty terms too weak",
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("model_diagnostics.png", dpi=150)
    plt.show()

    print("\nDiagnostic plots saved as 'model_diagnostics.png'")


def suggest_fixes():
    """Suggest specific fixes based on diagnosis"""

    print("\n" + "=" * 60)
    print("RECOMMENDED FIXES")
    print("=" * 60)

    print(
        """
1. IMMEDIATE FIX - Retrain with stronger directional balance:
   - Increase uniformity penalty from 0.05 to 0.2
   - Add explicit 50/50 directional target in loss
   - Use batch-level directional balance enforcement

2. DATA AUGMENTATION:
   - Synthetically create bearish scenarios
   - Oversample historical downturns
   - Add noise to prevent memorization

3. ARCHITECTURAL CHANGES:
   - Add dropout layers (0.2-0.3)
   - Reduce model capacity (fewer layers)
   - Use ensemble of models trained on different periods

4. LOSS FUNCTION REDESIGN:
   - Use ranking loss instead of MSE
   - Separate directional and magnitude predictions
   - Weight recent and historical data differently

5. VALIDATION STRATEGY:
   - Use rolling window validation
   - Test on 2022 bear market specifically
   - Ensure each validation batch has mixed directions
    """
    )


def main():
    print("\n" + "=" * 60)
    print("   COMPREHENSIVE MODEL DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Run all diagnostics
    checkpoint = load_model_and_analyze()
    analyze_prediction_patterns()
    analyze_training_data_distribution()
    create_diagnostic_plots()
    suggest_fixes()

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print(
        """
SUMMARY:
- Model learned to minimize loss by predicting small positive values
- This is safer than predicting negatives (markets trend up long-term)  
- The directional penalty in loss function was too weak
- Model found a local minimum: always predict slight gains

NEXT STEP:
Run the fixed training script with stronger penalties
or implement the backtesting anyway (55.9% accuracy is usable!)
    """
    )


if __name__ == "__main__":
    main()
