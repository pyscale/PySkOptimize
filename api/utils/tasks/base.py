import logging
import sys

from celery import Celery
from celery.signals import after_setup_logger

from ..settings import get_settings

celery_client = Celery(
    __name__,
    get_settings().CELERY_BROKER_URL,
    get_settings().CELERY_RESULT_BACKEND
)


@after_setup_logger.connect()
def logger_setup_handler(logger, **kwargs):
    my_handler = logging.StreamHandler(sys.stdout)

    my_handler.setLevel(logging.DEBUG)
    my_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # custom formatter
    my_handler.setFormatter(my_formatter)
    logger.addHandler(my_handler)

    logging.info("My log handler connected -> Global Logging")
