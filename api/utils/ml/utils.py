"""
This is the utility for all ML Model processes
"""
import importlib
from typing import Dict, List

import sklearn.compose
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skopt.searchcv import BayesSearchCV
from api.models import MLPipelineStateModel, SklearnTransformerModel, ParamType, \
    SklearnTransformerParamModel


def from_model_to_model(transformer_model: SklearnTransformerModel):
    """

    :param transformer_model:
    :return:
    """

    if "sklearn" in transformer_model.name:
        full_path = transformer_model.name
    else:
        full_path = f"sklearn.{transformer_model.name}"

    module_path = ".".join(full_path.split(".")[:-1])
    class_path = full_path.split(".")[-1]

    sklearn_module = importlib.import_module(module_path)

    model = getattr(sklearn_module, class_path)()

    # transformer_model.params

    return model


def from_model_to_model_params(transformer_params: List[SklearnTransformerParamModel]) -> Dict:
    """

    :param transformer_params:
    :return:
    """

    param_space = {}

    if transformer_params is None:
        return param_space

    for param in transformer_params:

        if param.type == ParamType.categorical:

            param_space[param.name] = param.boundValues

        elif param.type == ParamType.numeric:

            param_space[param.name] = (
                *param.boundValues,
                param.distribution
            )
        else:
            print("Warning")

    return param_space


def from_request_to_model(request: MLPipelineStateModel) -> BayesSearchCV:
    """

    :return:
    """

    res = []
    res_params = {}

    if request.targetTransformer is None:
        prefix = ""
    else:
        prefix = "regressor__"

    for pre_pod in request.preprocess:

        steps = []

        for i, transformer_model in enumerate(pre_pod.pipeline):

            step = from_model_to_model(transformer_model)

            if transformer_model.params is None:
                model_param = {}
            else:
                model_param = from_model_to_model_params(transformer_model.params)

            steps.append(
                (f'{pre_pod.name}_{i}', step)
            )

            res_params = {
                **res_params, **dict(
                    (f"{prefix}preprocess__{pre_pod.name}_{i}__{key}", value) for (key, value) in model_param.items()
                )
            }

        res.append(
            (
                pre_pod.name,
                Pipeline(
                    steps
                ),
                pre_pod.features
            )
        )

    if request.postprocess is None:
        base_model = Pipeline(
            [
                (
                    "preprocess", ColumnTransformer(res)
                ),
                (
                    "model", from_model_to_model(request.model)
                )
            ]
        )
    else:
        steps = []

        for i, transformer_model in enumerate(request.postprocess.pipeline):

            step = from_model_to_model(transformer_model)

            if transformer_model.params is None:
                model_param = {}
            else:
                model_param = from_model_to_model_params(transformer_model.params)

            steps.append(
                (f'step_{i}', step)
            )

            res_params = {
                **res_params, **dict(
                    (f"{prefix}postprocess__step_{i}__{key}", value) for (key, value) in model_param.items()
                )
            }

        base_model = Pipeline(
            [
                (
                    "preprocess", ColumnTransformer(res)
                ),
                (
                  "postprocess", Pipeline(
                      steps
                  )
                ),
                (
                    "model", from_model_to_model(request.model)
                )
            ]
        )

    res_params = {
        **res_params,
        **dict(
            (f"{prefix}model__{key}", value) for (key, value) in from_model_to_model_params(request.model.params).items()
        )
    }

    if request.targetTransformer is None:
        pass
    else:
        base_model = sklearn.compose.TransformedTargetRegressor(
            regressor=base_model,
            transformer=from_model_to_model(request.targetTransformer)
        )

    return BayesSearchCV(
        base_model,
        res_params,
        cv=5,
        scoring=request.scoring
    )
