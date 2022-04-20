"""
This is the utility for all ML Model processes
"""
import importlib
from typing import Dict

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skopt.searchcv import BayesSearchCV
from api.models import MLPipelineStateModel, SklearnTransformerModel, ParamType, SklearnTransformerParamModel


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


def from_model_to_model_params(transformer_params: Dict[str, SklearnTransformerParamModel]) -> Dict:
    """

    :param transformer_params:
    :return:
    """

    param_space = {}

    for params in list(transformer_params.keys()):

        if transformer_params[params].type == ParamType.integer:

            param_space[params] = (
                transformer_params[params].minValue,
                transformer_params[params].maxValue
            )
        elif transformer_params[params].type == ParamType.categorical:

            param_space[params] = transformer_params[params].categories

        elif transformer_params[params].type == ParamType.double:

            param_space[params] = (
                transformer_params[params].minValue,
                transformer_params[params].maxValue,
                transformer_params[params].distribution
            )
        else:
            print("Warning")

    return transformer_params


def from_request_to_model(request: MLPipelineStateModel) -> BayesSearchCV:
    """

    :return:
    """

    pre_pod_names = list(request.preprocess.keys())

    res = []
    res_params = {}
    for pre_pod_name in pre_pod_names:

        steps = []

        for i, transformer_model in enumerate(request.preprocess[pre_pod_name].pipeline):

            step = from_model_to_model(transformer_model)
            model_param = from_model_to_model_params(transformer_model.params)

            steps.append(
                (f'{pre_pod_name}_{i}', step)
            )

            res_params = {
                **res_params, **dict(
                    (f"preprocess__{pre_pod_name}_{i}__{key}", value) for (key, value) in model_param.items()
                )
            }

        res.append(
            (
                pre_pod_name,
                Pipeline(
                    steps
                ),
                [request.features[x].name for x in request.preprocess[pre_pod_name].features]
            )
        )

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

    res_params = {
        **res_params,
        **dict(
            (f"model__{key}", value) for (key, value) in from_model_to_model_params(request.model.params).items()
        )
    }

    return BayesSearchCV(
        base_model,
        res_params,
        cv=5,
        scoring=request.scoring
    )
