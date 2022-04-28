import io
from base64 import b64encode
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import get_scorer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from skopt.searchcv import BayesSearchCV

import plotly.express as px

from api.utils.ml.utils import from_request_to_model
from api.models import MLPipelineStateModel


@dataclass(frozen=True)
class _TrainingResults:

    trained_model: BayesSearchCV
    testing_score: float
    test_prediction: np.array
    test_actual: np.array
    validation_results: dict


def training_housing_model(model: BayesSearchCV) -> _TrainingResults:
    """

    :param model:
    :return:
    """
    cal_housing = fetch_california_housing()
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)

    model.fit(
        X_train,
        y_train
    )

    test_pred = model.predict(X_test)

    testing_score = model.score(X_test, y_test)

    return _TrainingResults(
        trained_model=model,
        test_prediction=test_pred,
        test_actual=y_test,
        testing_score=testing_score,
        validation_results=model.cv_results_
    )


def train_housing_demo(request_model: MLPipelineStateModel) -> str:
    """
    This is training an optimize model for the housing demo

    :return:
    """
    model = from_request_to_model(request_model)

    res = training_housing_model(
        model, request_model.scoring
    )

    buffer = io.StringIO()

    fig = px.scatter(
        x=res.test_actual,
        y=res.test_prediction,
        trendline="ols",
        title=f"Test Prediction Evaluations; {request_model.scoring} - {res.testing_score}"
    )

    fig.update_layout(
        xaxis_title="Actual Test Values",
        yaxis_title="Predicted Test Values"
    )

    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    return encoded
