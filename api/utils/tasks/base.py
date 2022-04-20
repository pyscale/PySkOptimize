from celery import Celery
from ..settings import get_settings

celery_client = Celery(
        __name__,
        get_settings().CELERY_BROKER_URL,
        get_settings().CELERY_RESULT_BACKEND
    )
