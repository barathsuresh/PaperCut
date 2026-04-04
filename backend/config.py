import os
from dotenv import load_dotenv

load_dotenv(override=True)

GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")
GCP_BUCKET_NAME: str = os.environ.get("GCP_BUCKET_NAME", "")
GCP_REGION: str = os.environ.get("GCP_REGION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS: str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "./gcp-key.json")

GEMINI_FLASH_MODEL: str = os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GEMINI_PRO_MODEL: str = os.environ.get("GEMINI_PRO_MODEL", "gemini-2.5-pro")

_REQUIRED = ["GCP_PROJECT_ID", "GCP_BUCKET_NAME"]


def validate() -> None:
    """Raise RuntimeError listing every missing required config value."""
    missing = [k for k in _REQUIRED if not globals().get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {missing}. "
            "Check your .env file before starting the server."
        )
