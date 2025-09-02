"""
Central configuration management
"""

from pathlib import Path
from typing import List

from src.config.secrets import secrets


class Config:
    """Project configuration"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CACHE_DIR = DATA_DIR / "cache"
    MODELS_DIR = PROJECT_ROOT / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

    # API Keys (from Doppler)
    ALPHA_VANTAGE_KEY = secrets.get("ALPHA_VANTAGE_API_KEY")
    NEWSAPI_KEY = secrets.get("NEWSAPI_API_KEY")
    HUGGINGFACE_TOKEN = secrets.get("HUGGINGFACE_API_KEY")
    WANDB_KEY = secrets.get("WANDB_API_KEY")

    # Data Collection
    DEFAULT_TICKERS: List[str] = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "JPM",
        "V",
        "JNJ",
    ]
    YEARS_OF_DATA = 5

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = "cuda" if secrets.get("DEVICE", "cuda") == "cuda" else "cpu"

    # Model
    SEQUENCE_LENGTH = 60
    FORECAST_HORIZON = 5
    NUM_FEATURES = 7

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINTS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Initialize directories
Config.create_directories()
