"""
app/config.py
─────────────
Configuration loader and CPU optimization settings for Medical RAG Assistant.
"""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Set CPU threading environment variables at the top to prevent oversubscription
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Load environment variables (.env)
load_dotenv()

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to config.yaml
CONFIG_PATH = BASE_DIR / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    settings: dict = yaml.safe_load(f) or {}

__all__ = ["settings", "BASE_DIR"]
