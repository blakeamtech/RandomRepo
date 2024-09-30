from pydantic_settings import BaseSettings
import dotenv

dotenv.load_dotenv()

class Settings(BaseSettings):
    AUTH_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()