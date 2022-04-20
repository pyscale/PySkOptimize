from celery import Celery
from ..settings import Settings


celery_client = Celery(
        __name__,
        broker,
        backend
    )
