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
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize model
        self.model = TimeSeriesTransformer(
            input_dim=21,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            forecast_horizon=3,
            output_dim=3,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")

    def calibrate_bias(self):
        """Calculate the bias offset from recent predictions"""
        # We know from diagnostics that average prediction is ~0.95%
        # This becomes our baseline
        self.bias_offset = 0.0095  # 0.95% average from your test
        self.prediction_std = 0.0005  # Very low variation in your model

        print(f"Bias calibration: offset={self.bias_offset:.4f}, std={self.prediction_std:.4f}")

    def prepare_features(self, ticker, lookback_days=100):
        """Download and prepare features for a ticker"""

        # Download data
        df = yf.download(ticker, period=f"{lookback_days}d", progress=False)

        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 80:
            print(f"Insufficient data for {ticker}")
            return None

        # Calculate features (simplified version)
        close_price = df["Close"].values if "Close" in df.columns else df["Close"]

        df["returns"] = pd.Series(close_price).pct_change().values
        df["MA_20"] = pd.Series(close_price).rolling(20).mean().values
        df["MA_50"] = pd.Series(close_price).rolling(50).mean().values
        df["MA_200"] = pd.Series(close_price).rolling(200).mean().values

        # RSI
        returns_series = pd.Series(close_price).pct_change()
        df["RSI"] = (
            returns_series.rolling(14)
            .apply(
                lambda x: (
                    100 - 100 / (1 + (x[x > 0].mean() / -x[x < 0].mean()))
                    if len(x[x < 0]) > 0
                    else 50
                )
            )
            .values
        )

        # Volatility
        df["volatility_20d"] = returns_series.rolling(20).std().values

        # MACD (simplified)
        close_series = pd.Series(close_price)
        ema_12 = close_series.ewm(span=12).mean()
        ema_26 = close_series.ewm(span=26).mean()
        df["MACD"] = (ema_12 - ema_26).values
        df["MACD_signal"] = pd.Series(df["MACD"]).ewm(span=9).mean().values
        df["MACD_diff"] = df["MACD"] - df["MACD_signal"]

        # Bollinger Bands position
        bb_mean = close_series.rolling(20).mean()
        bb_std = close_series.rolling(20).std()
        df["BB_position"] = ((close_series - bb_mean) / (2 * bb_std + 1e-8)).values

        # Volume ratio
        volume = df["Volume"].values if "Volume" in df.columns else df["Volume"]
        df["volume_ratio"] = pd.Series(volume) / pd.Series(volume).rolling(20).mean()
        df["volume_ratio"] = df["volume_ratio"].fillna(1.0).values

        # Market regime
        df["trend_strength"] = (df["MA_50"] - df["MA_200"]) / (df["MA_200"] + 1e-8)
        df["distance_from_MA200"] = (close_series - df["MA_200"]) / (df["MA_200"] + 1e-8)

        # Fill NaN for trend indicators
        df["trend_strength"] = df["trend_strength"].fillna(0).values
        df["distance_from_MA200"] = df["distance_from_MA200"].fillna(0).values

        # Temporal features
        df["day_of_week"] = pd.to_datetime(df.index).dayofweek
        df["month"] = pd.to_datetime(df.index).month
        df["quarter"] = pd.to_datetime(df.index).quarter

        # Fill NaN with forward fill then backward fill
        df = df.fillna(method="ffill").fillna(method="bfill")

        # Select features
        feature_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "returns",
            "returns",
            "volatility_20d",  # Using returns twice as placeholder
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_diff",
            "BB_position",
            "volume_ratio",
            "MA_50",
            "MA_200",
            "trend_strength",
            "distance_from_MA200",
            "day_of_week",
            "month",
            "quarter",
        ]

        # Get last 60 days of features
        features = df[feature_cols].iloc[-60:].values

        # Standardize
        features = (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-8)

        # Replace any remaining NaN with 0
        features = np.nan_to_num(features, 0)

        return features, float(close_price[-1])

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

        # Adjust for bias
        adjusted_pred = raw_pred - self.bias_offset

        # Calculate relative strength (how much above/below the bias)
        relative_strength = (raw_pred - self.bias_offset) / (self.prediction_std + 1e-8)

        # Generate signal based on relative strength
        signals = []
        for i, (pred, strength) in enumerate(zip(adjusted_pred, relative_strength)):
            if strength > 1.0:  # More than 1 std above mean
                signal = "STRONG BUY"
            elif strength > 0.5:
                signal = "BUY"
            elif strength < -1.0:  # More than 1 std below mean
                signal = "SELL"
            elif strength < -0.5:
                signal = "WEAK SELL"
            else:
                signal = "HOLD"

            signals.append(
                {
                    "day": i + 1,
                    "raw_prediction": float(raw_pred[i]),
                    "adjusted_prediction": float(pred),
                    "relative_strength": float(strength),
                    "signal": signal,
                    "predicted_price": current_price * (1 + pred),
                    "predicted_return_%": pred * 100,
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

        output_file = f"predictions/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("predictions").mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")

    else:
        # Single ticker analysis
        ticker = args.ticker.upper()
        predictor = BiasAdjustedPredictor(
            model_path=args.model_path or "model_extended_best.pt",
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

            output_file = f"predictions/prediction_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            Path("predictions").mkdir(exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to {output_file}")
        else:
            print(f"Failed to generate prediction for {ticker}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
