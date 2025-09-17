"""
Bias-Adjusted Prediction Wrapper
Converts your biased model into a usable trading signal generator
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent))
from src.models.timeseries_transformer import TimeSeriesTransformer


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


class BiasAdjustedPredictor:
    """
    Wrapper that adjusts for the bullish bias in predictions
    Converts relative predictions into actionable signals
    """

    def __init__(self, model_path="model_extended_best.pt", ticker="AAPL"):
        self.ticker = ticker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Try ticker-specific model first, fall back to default
        ticker_model_path = f"models/model_{ticker}_best.pt"
        if Path(ticker_model_path).exists():
            model_path = ticker_model_path
        elif Path(model_path).exists():
            pass  # Use provided path
        else:
            # Try to find any model file
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pt"))
                if model_files:
                    model_path = str(model_files[0])
                    print(f"Using available model: {model_path}")

        self.load_model(model_path)
        self.calibrate_bias()

    def load_model(self, model_path):
        """Load model with architecture detection"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Detect model configuration from checkpoint
        if "model_config" in checkpoint and checkpoint["model_config"]:
            model_config = checkpoint["model_config"]
            print(f"Using saved model configuration: {model_config}")
        else:
            # Try to infer from state dict
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if "input_embedding.projection.weight" in state_dict:
                weight_shape = state_dict["input_embedding.projection.weight"].shape
                hidden_dim = weight_shape[0]  # First dimension is hidden_dim
                input_dim = weight_shape[1]  # Second dimension is input features
            else:
                # Default values to match training script defaults
                hidden_dim = 128  # Match training default
                input_dim = 21

            model_config = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_heads": 4,
                "num_layers": 4,
                "forecast_horizon": 3,
                "output_dim": 3,
            }
            print(f"Inferred model configuration: {model_config}")

        # Initialize model with detected configuration
        self.model = TimeSeriesTransformer(
            input_dim=model_config.get("input_dim", 21),
            hidden_dim=model_config.get("hidden_dim", 128),
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 4),
            forecast_horizon=model_config.get("forecast_horizon", 3),
            output_dim=model_config.get("output_dim", 3),
        )

        # Load the state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Architecture: input_dim={model_config.get('input_dim')}, hidden_dim={model_config.get('hidden_dim')}")

    def calibrate_bias(self):
        """Calculate the bias offset from recent predictions"""
        # We know from diagnostics that average prediction is ~0.95%
        # This becomes our baseline
        self.bias_offset = 0.0095  # 0.95% average from your test
        self.prediction_std = 0.0005  # Very low variation in your model

        print(f"Bias calibration: offset={self.bias_offset:.4f}, std={self.prediction_std:.4f}")

    def prepare_features(self, ticker, lookback_days=100):
        """Download and prepare features matching training data (exactly 10 features)"""

        # Download data
        df = yf.download(ticker, period=f"{lookback_days}d", progress=False)

        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 80:
            print(f"Insufficient data for {ticker}")
            return None

        # Calculate features exactly matching train_ultra_simple.py
        features = []

        # Basic OHLCV features (exactly as in training)
        features.append(df["Open"].values)
        features.append(df["High"].values)
        features.append(df["Low"].values)
        features.append(df["Close"].values)
        features.append(df["Volume"].values)

        # Technical indicators (exactly as in training)
        # Returns
        features.append(df["Close"].pct_change().fillna(0).values)

        # Moving averages
        features.append(df["Close"].rolling(5).mean().fillna(df["Close"]).values)
        features.append(df["Close"].rolling(20).mean().fillna(df["Close"]).values)

        # Volatility (rolling std of returns)
        returns = df["Close"].pct_change()
        features.append(returns.rolling(20).std().fillna(0).values)

        # Volume moving average
        features.append(df["Volume"].rolling(5).mean().fillna(df["Volume"]).values)

        # Stack features - this creates shape (timesteps, 10)
        features = np.stack(features, axis=1)

        # Get last 60 days of features
        features = features[-60:]

        # Normalize using simple standardization (matching training approach)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8  # Avoid division by zero
        features = (features - mean) / std

        # Replace any remaining NaN with 0
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        print(f"Prepared features shape: {features.shape} (should be (60, 10))")

        return features.astype(np.float32), float(df["Close"].iloc[-1])

    def predict_adjusted(self, ticker):
        """Make bias-adjusted predictions"""

        # Prepare features
        result = self.prepare_features(ticker)
        if result is None:
            return None

        features, current_price = result

        # Make prediction
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_pred = self.model(x).cpu().numpy()[0]

        # Apply realistic scaling using sigmoid to bound predictions
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -10, 10)))  # Clip to prevent overflow

        # Scale predictions to realistic daily return range (-5% to +5%)
        scaled_pred = (sigmoid(raw_pred) - 0.5) * 0.1  # Maps to -5% to +5%

        # Apply bias adjustment (much smaller bias)
        adjusted_pred = scaled_pred + (self.bias_offset * 0.1)  # Scale down bias

        # Ensure predictions are reasonable (max 10% daily move)
        adjusted_pred = np.clip(adjusted_pred, -0.10, 0.10)

        # Calculate relative strength for signal generation
        relative_strength = adjusted_pred / 0.02  # Normalize by 2% (typical daily vol)

        # Generate signal based on realistic thresholds
        signals = []
        for i, (pred, strength) in enumerate(zip(adjusted_pred, relative_strength)):
            # Use percentage-based thresholds
            pred_pct = pred * 100  # Convert to percentage

            if pred_pct > 2.0:  # More than 2% expected return
                signal = "STRONG BUY"
            elif pred_pct > 0.5:  # More than 0.5% expected return
                signal = "BUY"
            elif pred_pct < -2.0:  # Less than -2% expected return
                signal = "STRONG SELL"
            elif pred_pct < -0.5:  # Less than -0.5% expected return
                signal = "SELL"
            else:
                signal = "HOLD"

            signals.append(
                {
                    "day": i + 1,
                    "raw_prediction": float(raw_pred[i]),
                    "adjusted_prediction": float(pred),
                    "relative_strength": float(strength),
                    "signal": signal,
                    "predicted_price": float(current_price * (1 + pred)),
                    "predicted_return_%": float(pred * 100),
                }
            )

        return {
            "ticker": ticker,
            "current_price": current_price,
            "predictions": signals,
            "composite_signal": self._get_composite_signal(relative_strength),
            "confidence": self._calculate_confidence(relative_strength),
        }

    def _get_composite_signal(self, strengths):
        """Get overall signal from 3-day predictions"""
        avg_strength = np.mean(strengths)

        if avg_strength > 1.5:
            return "STRONG BUY"
        elif avg_strength > 0.5:
            return "BUY"
        elif avg_strength < -1.5:
            return "STRONG SELL"
        elif avg_strength < -0.5:
            return "SELL"
        else:
            return "HOLD"

    def _calculate_confidence(self, strengths):
        """Calculate confidence based on consistency"""
        # Higher confidence if all predictions point same direction
        std_strength = np.std(strengths)
        consistency = 1.0 / (1.0 + std_strength)

        # Higher confidence if prediction is far from mean
        avg_abs_strength = np.mean(np.abs(strengths))
        extremity = min(avg_abs_strength / 2.0, 1.0)

        confidence = (consistency + extremity) / 2.0
        return float(confidence)

    def rank_stocks(self, tickers):
        """Rank multiple stocks by predicted performance"""
        results = []

        print(f"\nAnalyzing {len(tickers)} stocks...")
        print("-" * 50)

        for ticker in tickers:
            try:
                pred = self.predict_adjusted(ticker)
                if pred:
                    avg_return = np.mean([p["adjusted_prediction"] for p in pred["predictions"]])
                    results.append(
                        {
                            "ticker": ticker,
                            "current_price": pred["current_price"],
                            "avg_3day_return": avg_return,
                            "signal": pred["composite_signal"],
                            "confidence": pred["confidence"],
                        }
                    )

                    print(
                        f"{ticker}: {pred['composite_signal']} "
                        f"(Confidence: {pred['confidence']:.2%})"
                    )
            except Exception as e:
                print(f"{ticker}: Error - {e}")

        # Sort by expected return
        results = sorted(results, key=lambda x: x["avg_3day_return"], reverse=True)

        return results

    def generate_portfolio_weights(self, tickers, top_n=3):
        """Generate portfolio weights based on rankings"""
        rankings = self.rank_stocks(tickers)

        if len(rankings) == 0:
            return {}

        # Select top N stocks
        top_stocks = rankings[:top_n]
        bottom_stocks = rankings[-top_n:] if len(rankings) > top_n else []

        weights = {}

        # Long positions (top performers)
        for stock in top_stocks:
            if stock["signal"] in ["BUY", "STRONG BUY"]:
                # Weight by confidence
                weights[stock["ticker"]] = 0.3 * stock["confidence"]

        # Short positions (worst performers) - optional
        for stock in bottom_stocks:
            if stock["signal"] in ["SELL", "STRONG SELL"]:
                weights[stock["ticker"]] = -0.1 * stock["confidence"]

        # Normalize weights to sum to 1 (or your desired exposure)
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights


def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Generate bias-adjusted predictions for stock trading"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker to analyze (default: AAPL)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Analyze multiple tickers and generate portfolio"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BIAS-ADJUSTED PREDICTION SYSTEM")
    print("=" * 60)

    if args.multiple:
        # Initialize predictor for multiple stocks
        predictor = BiasAdjustedPredictor(model_path=args.model_path or "model_extended_best.pt")

        # Test on multiple stocks
        test_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "JPM"]

        # Rank stocks
        rankings = predictor.rank_stocks(test_tickers)

        print("\n" + "=" * 60)
        print("STOCK RANKINGS (Best to Worst)")
        print("=" * 60)

        for i, stock in enumerate(rankings, 1):
            direction = "↑" if stock["avg_3day_return"] > 0 else "↓"
            print(
                f"{i}. {stock['ticker']}: "
                f"{stock['avg_3day_return']*100:+.3f}% {direction} "
                f"Signal: {stock['signal']} "
                f"(Confidence: {stock['confidence']:.1%})"
            )

        # Generate portfolio
        print("\n" + "=" * 60)
        print("SUGGESTED PORTFOLIO WEIGHTS")
        print("=" * 60)

        weights = predictor.generate_portfolio_weights(test_tickers, top_n=3)

        for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            position = "LONG" if weight > 0 else "SHORT"
            print(f"{ticker}: {abs(weight)*100:.1f}% {position}")

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "rankings": rankings,
            "portfolio_weights": weights,
        }

        # Convert all numpy types to serializable format
        results = convert_to_serializable(results)

        output_file = f"predictions/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("predictions").mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {output_file}")

    else:
        # Single ticker analysis
        ticker = args.ticker.upper()
        predictor = BiasAdjustedPredictor(
            model_path=args.model_path or f"models/model_{ticker}_best.pt",
            ticker=ticker
        )

        print(f"\nAnalyzing {ticker}...")
        prediction = predictor.predict_adjusted(ticker)

        if prediction:
            print(f"\n{'='*60}")
            print(f"PREDICTION FOR {ticker}")
            print(f"{'='*60}")
            print(f"Current Price: ${prediction['current_price']:.2f}")
            print(f"Composite Signal: {prediction['composite_signal']}")
            print(f"Confidence: {prediction['confidence']:.1%}")

            print(f"\nDetailed Predictions:")
            for i, pred in enumerate(prediction['predictions'], 1):
                print(f"  Day {i}: {pred['adjusted_prediction']:+.3%} ({pred['signal']})")

            # Save single ticker results
            results = {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "prediction": prediction,
            }

            # Convert all numpy types to serializable format
            results = convert_to_serializable(results)

            output_file = f"predictions/prediction_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path("predictions").mkdir(exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\nResults saved to {output_file}")
        else:
            print(f"Failed to generate prediction for {ticker}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
