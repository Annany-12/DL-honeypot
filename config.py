"""Configuration settings for the honeypot simulator."""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# AI Model Configuration
GPT2_MODEL_NAME = os.getenv("GPT2_MODEL_NAME", "gpt2")  # Using small GPT-2 (124M parameters)
GPT2_MAX_LENGTH = int(os.getenv("GPT2_MAX_LENGTH", "512"))
GPT2_TEMPERATURE = float(os.getenv("GPT2_TEMPERATURE", "0.8"))
GPT2_MAX_NEW_TOKENS = int(os.getenv("GPT2_MAX_NEW_TOKENS", "150"))
GPT2_DO_SAMPLE = os.getenv("GPT2_DO_SAMPLE", "true").lower() == "true"
GPT2_TOP_P = float(os.getenv("GPT2_TOP_P", "0.9"))
GPT2_TOP_K = int(os.getenv("GPT2_TOP_K", "50"))

# CTGAN Configuration
CTGAN_EPOCHS = int(os.getenv("CTGAN_EPOCHS", "50"))
CTGAN_BATCH_SIZE = int(os.getenv("CTGAN_BATCH_SIZE", "500"))
CTGAN_GENERATOR_DIM = tuple(map(int, os.getenv("CTGAN_GENERATOR_DIM", "256,256").split(",")))
CTGAN_DISCRIMINATOR_DIM = tuple(map(int, os.getenv("CTGAN_DISCRIMINATOR_DIM", "256,256").split(",")))
CTGAN_LEARNING_RATE = float(os.getenv("CTGAN_LEARNING_RATE", "2e-4"))

# TimeGAN Configuration
TIMEGAN_SEQ_LEN = int(os.getenv("TIMEGAN_SEQ_LEN", "24"))
TIMEGAN_N_SEQ = int(os.getenv("TIMEGAN_N_SEQ", "5"))
TIMEGAN_HIDDEN_DIM = int(os.getenv("TIMEGAN_HIDDEN_DIM", "24"))
TIMEGAN_GAMMA = float(os.getenv("TIMEGAN_GAMMA", "1"))
TIMEGAN_BATCH_SIZE = int(os.getenv("TIMEGAN_BATCH_SIZE", "32"))
TIMEGAN_LEARNING_RATE = float(os.getenv("TIMEGAN_LEARNING_RATE", "5e-4"))

# LSTM Fallback Configuration (when TimeGAN is not available)
LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "64"))
LSTM_NUM_LAYERS = int(os.getenv("LSTM_NUM_LAYERS", "2"))
LSTM_LEARNING_RATE = float(os.getenv("LSTM_LEARNING_RATE", "0.001"))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "50"))

# Honeypot Configuration
SSH_PORT = 2222
FAKE_HOSTNAME = "honeypot-server"
FAKE_USERNAME = "admin"
FAKE_PASSWORD = "admin123"

# Dashboard Configuration
DASHBOARD_PORT = 8501
API_PORT = 8000

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
