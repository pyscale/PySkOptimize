import logging
import sys
from typing import Dict

from celery import Celery
from celery.signals import after_setup_logger

from api.utils.ml.train import train_housing_demo
from api.models import MLPipelineStateModel
from .settings import get_settings

settings = get_settings()

celery_client = Celery(
    __name__,
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)


@after_setup_logger.connect()
def logger_setup_handler(logger, **kwargs):
    my_handler = logging.StreamHandler(sys.stdout)

    my_handler.setLevel(logging.DEBUG)
    my_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # custom formatter
    my_handler.setFormatter(my_formatter)
    logger.addHandler(my_handler)

    logging.info("My log handler connected -> Global Logging")


@celery_client.task(name="housing_demo")
def task_housing_demo(request_model: Dict) -> str:
    """

    :return:
    """

    return train_housing_demo(
        MLPipelineStateModel.parse_obj(request_model)
    )
