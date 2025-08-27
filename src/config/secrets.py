"""
Secrets management via Doppler
"""
import os
import subprocess
import json
from typing import Dict, Any

class SecretsManager:
    """Manage secrets from Doppler"""
    
    def __init__(self):
        self.secrets = self._load_secrets()
    
    def _load_secrets(self) -> Dict[str, Any]:
        """Load secrets from Doppler or environment"""
        try:
            # Try to get secrets from Doppler
            result = subprocess.run(
                ['doppler', 'secrets', 'download', '--no-file', '--format', 'json'],
                capture_output=True,
                text=True,
                check=True
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to environment variables
            return {
                'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY', ''),
                'NEWSAPI_API_KEY': os.getenv('NEWSAPI_API_KEY', ''),
                'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY', ''),
                'WANDB_API_KEY': os.getenv('WANDB_API_KEY', ''),
                'DB_URL': os.getenv('DB_URL', 'sqlite:///timeseries.db'),
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get secret value"""
        return self.secrets.get(key, default)

# Global instance
secrets = SecretsManager()