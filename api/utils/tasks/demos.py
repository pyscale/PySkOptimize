import io
from base64 import b64encode

import pandas as pd

from sklearn.metrics import get_scorer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import plotly.express as px

from api.utils.ml.utils import from_request_to_model
from api.utils.ml.train import train_housing_demo

from api.models import MLPipelineStateModel

from .base import celery_client


@celery_client.task(name="housing_demo")
def task_housing_demo(request_model: MLPipelineStateModel) -> str:
    """

    :return:
    """
    return train_housing_demo(request_model)