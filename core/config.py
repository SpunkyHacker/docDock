"""
System configuration, model identifiers, default hyperparameters, and device resolution.
Loads environment variables from .env without exposing secrets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables from .env file if present
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Local & Cloud Model Identifiers
BI_ENCODER_ID = "NeuML/pubmedbert-base-embeddings"
CROSS_ENCODER_ID = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Algorithm Hyperparameters
BI_ENCODER_TEMPERATURE = 12.0
TEXT_CROSS_WEIGHT = 0.55
TEXT_BI_WEIGHT = 0.45
IMAGE_CLIP_WEIGHT = 0.50
IMAGE_VISION_TEXT_WEIGHT = 0.50
IMAGE_TEMPERATURE = 0.3  # Multiplier on BiomedCLIP logit_scale to prevent saturation
MAX_TEXT_WORDS = 350

# General Medicine & Fallback Routing Configuration
GENERAL_MEDICINE_DEPT = "General Medicine (Internal Medicine)"
LOW_CONFIDENCE_THRESHOLD = 0.35
AMBIGUITY_MARGIN = 0.08
MIN_MEANINGFUL_WORDS = 3

# Learning-to-Rank (LTR) Doctor Ranking Weights
LTR_W_CONF = 0.30
LTR_W_TAG = 0.25
LTR_W_SUCCESS = 0.15
LTR_W_RATING = 0.10
LTR_W_EXP = 0.10
LTR_W_AVAIL = 0.10
LTR_EXP_MAX_YEARS = 25.0

# Device Resolution
def get_torch_device() -> torch.device:
    """
    Auto-detect hardware acceleration (CUDA -> MPS -> CPU).
    Returns torch.device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_pipeline_device():
    """
    Returns the appropriate device parameter for HuggingFace pipeline.
    0 for CUDA GPU 0, "mps" for Apple Silicon MPS, -1 for CPU.
    """
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1

def get_groq_api_key(override_key: str = "") -> str:
    """
    Safely retrieves the active Groq API key (UI override takes precedence over .env).
    Does not log or print the key.
    """
    if override_key and override_key.strip():
        return override_key.strip()
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    return os.getenv("GROQ_API_KEY", "").strip()
