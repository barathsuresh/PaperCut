import os

# Set environment variables before any backend modules are imported
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_BUCKET_NAME", "test-bucket")
os.environ.setdefault("GCP_REGION", "us-central1")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")
