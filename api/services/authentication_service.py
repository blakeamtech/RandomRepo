"""
Class to determine if the authentication key of a user is valid.
"""

from fastapi import HTTPException, status
from api.core.config import settings
import hashlib

class AuthenticationService:
    def __init__(self, auth_key: str):
        self.auth_key = auth_key

    def _hash_key(self, key: str) -> str:
        """
        Hash the given key using SHA-256.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def is_valid(self) -> bool:
        """
        Check if the authentication key is valid.
        """
        hashed_input = self._hash_key(self.auth_key)
        return hashed_input == settings.AUTH_KEY

    def raise_exception_if_invalid(self):
        """
        Raise an exception if the authentication key is invalid.
        """
        if not self.is_valid():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication key"
            )
        return True