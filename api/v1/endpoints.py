import logging
from api.services.inference_service import InferenceService
from api.services.authentication_service import AuthenticationService
from fastapi import APIRouter, HTTPException, Query, Body

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/inference")
async def inference(
    input_context: str,  # Expecting a single string input
    auth_key: str,
    max_length: int = Query(512),  # Allow custom max_length via the request
    temperature: float = Query(0.3)  # Allow custom temperature via the request
):
    """
    Endpoint for single inference (batch size 1), where user submits a single string as input.
    """
    try:
        # Initialize the inference service
        inference_service = InferenceService(auth_key)
        # Validate the auth_key
        AuthenticationService(auth_key).raise_exception_if_invalid()

        # Call the generate_text method for a single input context (batch size 1)
        generated_text = inference_service.generate_text(input_context, max_length=max_length, temperature=temperature)
        return {"generated_text": generated_text}

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")


@router.post("/batch_inference")
async def batch_inference(
    input_context: str,  # A single input string
    auth_key: str,
    num_batches: int = Query(1),  # Number of times to duplicate the input to simulate batch size
    max_length: int = Query(128),
    temperature: float = Query(0.7)
):
    """
    Endpoint for batch inference, where the user submits a single string that will be duplicated.
    """
    try:
        # Initialize the inference service
        inference_service = InferenceService(auth_key)
        # Validate the auth_key
        AuthenticationService(auth_key).raise_exception_if_invalid()

        # Duplicate the single input context for batch processing
        input_contexts = [input_context] * num_batches

        # Call the batch generate_text_with_batch_size method
        generated_texts = inference_service.generate_text_with_batch_size(
            input_contexts, batch_size=num_batches, max_length=max_length, temperature=temperature
        )
        return {"generated_texts": generated_texts}

    except Exception as e:
        logger.error(f"Error during batch inference: {e}")
        raise HTTPException(status_code=500, detail=f"Batch text generation failed: {str(e)}")


@router.get("/login")
async def login(
    auth_key: str,
):
    """
    Endpoint for login validation.
    """
    try:
        authentication_service = AuthenticationService(auth_key)
        print({'authenticated' : authentication_service.is_valid()})
        return {'authenticated' : authentication_service.is_valid()}

    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")