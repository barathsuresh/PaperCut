import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv_path = PROJECT_ROOT / ".env"
dotenv_example_path = PROJECT_ROOT / ".env.example"

if dotenv_path.exists():
    load_dotenv(dotenv_path)
elif dotenv_example_path.exists():
    load_dotenv(dotenv_example_path)
else:
    load_dotenv()

GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")
GCP_BUCKET_NAME: str = os.environ.get("GCP_BUCKET_NAME", "")
GCP_REGION: str = os.environ.get("GCP_REGION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS: str = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "./gcp-key.json")

GEMINI_FLASH_MODEL: str = os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GEMINI_PRO_MODEL: str = os.environ.get("GEMINI_PRO_MODEL", "gemini-2.5-pro")


class ConfigError(ValueError):
    pass


def runtime_environment() -> dict[str, str]:
    return {
        "GCP_PROJECT_ID": GCP_PROJECT_ID,
        "GCP_BUCKET_NAME": GCP_BUCKET_NAME,
        "GCP_REGION": GCP_REGION,
        "GOOGLE_APPLICATION_CREDENTIALS": GOOGLE_APPLICATION_CREDENTIALS,
        "GEMINI_FLASH_MODEL": GEMINI_FLASH_MODEL,
        "GEMINI_PRO_MODEL": GEMINI_PRO_MODEL,
    }


def validate_runtime_config(*, requires_bucket: bool = False) -> None:
    missing = []

    if not GCP_PROJECT_ID:
        missing.append("GCP_PROJECT_ID")
    if not GOOGLE_APPLICATION_CREDENTIALS:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if requires_bucket and not GCP_BUCKET_NAME:
        missing.append("GCP_BUCKET_NAME")

    if missing:
        joined = ", ".join(missing)
        raise ConfigError(
            f"Missing required environment variables: {joined}. "
            "Copy .env.example to .env and set your GCP values."
        )


def validate() -> None:
    validate_runtime_config(requires_bucket=True)
