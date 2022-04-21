import json

import pytest

from fastapi.testclient import TestClient

from api.models import MLPipelineStateModel, FeaturePodModel, \
    SklearnTransformerParamModel, SklearnTransformerModel, \
    DistributionEnum, ParamType

from api.__main__ import api


@pytest.fixture
def demo_simple_housing():
    """

    :return:
    """

    return MLPipelineStateModel(
        model=SklearnTransformerModel(
            name="sklearn.linear_model.Ridge",
            params=[
                SklearnTransformerParamModel(
                    name="alpha",
                    type=ParamType.double,
                    minValue=1e-16,
                    maxValue=1e16,
                    distribution=DistributionEnum.log_uniform
                )
            ]
        ),
        scoring="neg_mean_squared_error",
        preprocess=[
            FeaturePodModel(
                name="featurePod1",
                features=[
                    'MedInc',
                    'HouseAge',
                    'AveRooms',
                    'Population',
                    'AveOccup',
                    'Latitude',
                    'Longitude'
                ],
                pipeline=[
                    SklearnTransformerModel(
                        name="sklearn.preprocessing.PowerTransformer",
                    ),
                ]
            )
        ],
        postprocess=None
    )


@pytest.fixture
def client():
    return TestClient(api)


@pytest.fixture
def housing_ml_pipeline_state():

    with open("/usr/src/app/data/test.json", "r") as f:
        data = json.load(f)

    return data
