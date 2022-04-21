import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    """

    """

    CELERY_BROKER_URL: str = os.environ.get("CELERY_BROKER_URL", "amqp://")  # NEW
    CELERY_RESULT_BACKEND: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://")  # NEW
