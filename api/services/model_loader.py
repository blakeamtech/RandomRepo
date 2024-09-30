import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    try:
        model_dir = os.getenv('MODEL_DIR', '/workspace/model')
        logger.info(f"Loading custom model from {model_dir} on device: {device}")

        # Clear GPU memory before loading the model
        torch.cuda.empty_cache()

        # Load model using FP16 (half precision) for memory savings
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True
        ).half().to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Enable gradient checkpointing for additional memory efficiency
        model.gradient_checkpointing_enable()

        # Clear any residual memory after model loading
        torch.cuda.empty_cache()

        logger.info("Model and tokenizer successfully loaded with optimizations.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading custom model or tokenizer: {e}", exc_info=True)
        raise RuntimeError(f"Error loading custom model or tokenizer: {str(e)}")

# Load the model and tokenizer
model, tokenizer = load_model()