import json
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("scalers")
OUT.mkdir(parents=True, exist_ok=True)

# Choose the columns your model expects
FEATURES = ["Open", "High", "Low", "Close", "Volume"]


def make_scaler(df: pd.DataFrame):
    stats = {}
    for col in FEATURES:
        if col not in df.columns:
            continue
        s = df[col].astype("float64")
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0) if s.std(ddof=0) else 1.0),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return {
        "created": datetime.now().isoformat(),
        "features": FEATURES,
        "stats": stats,
        "method": "zscore",  # informational
    }


def main():
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        print("No parquet files found in data/raw")
        return 1

    made = 0
    for f in files:
        ticker = f.stem.upper()
        out = OUT / f"scaler_{ticker}.json"
        if out.exists():
            print(f"[skip] {out.name} exists")
            continue
        df = pd.read_parquet(f)
        scaler = make_scaler(df)
        with open(out, "w") as w:
            json.dump({"ticker": ticker, **scaler}, w, indent=2)
        made += 1
        print(f"[ok]   wrote {out.name}")
    print(f"\nDone. Created {made} scaler file(s) in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
