from api.services.authentication_service import AuthenticationService
from api.services.inference_service import InferenceService
from fastapi import Depends

def get_authentication_service(auth_key: str = Depends(AuthenticationService)):
    return AuthenticationService(auth_key)

def get_inference_service(auth_key: str = Depends(get_authentication_service)):
    return InferenceService(auth_key)