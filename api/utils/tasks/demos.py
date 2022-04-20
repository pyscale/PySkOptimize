import io
from base64 import b64encode

import pandas as pd

from sklearn.metrics import get_scorer
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import plotly.express as px

from api.utils.ml.utils import from_request_to_model
from api.models import MLPipelineStateModel

from .base import celery_client


@celery_client.task
def task_housing_demo(request_model: MLPipelineStateModel) -> str:
    """

    :return:
    """
    model = from_request_to_model(request_model)

    cal_housing = fetch_california_housing()
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    y -= y.mean()

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)

    model.fit(
        X_train,
        y_train
    )

    scorer_funct = get_scorer(request_model.scoring)

    test_pred = model.predict(X_test)

    testing_score = scorer_funct(y_test, test_pred)

    buffer = io.StringIO()

    fig = px.scatter(
        x=y_test,
        y=test_pred,
        trendline="ols",
        title=f"Test Prediction Evaluations; {request_model.scoring} - {testing_score}"
    )

    fig.update_layout(
        xaxis_title="Actual Test Values",
        yaxis_title="Predicted Test Values"
    )

    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    return encoded
