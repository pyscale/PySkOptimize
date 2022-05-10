from enum import Enum
import importlib

from typing import Dict, List, Union, Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from skopt.searchcv import BayesSearchCV

Numeric = Union[float, int]


@dataclass(frozen=True)
class _ColumnTransformerInput:
    """
    This is a private class to handle raw input

    """
    name: str
    sk_obj: Pipeline
    features: List[str]

    def to_raw(self):
        if self.features is None:
            return (
                self.name,
                self.sk_obj
            )
        else:
            return (
                self.name,
                self.sk_obj,
                self.features
            )


class DistributionEnum(str, Enum):
    """
    This is for enumeration
    """
    normal: str = "normal"
    log_normal: str = "log-normal"
    uniform: str = "uniform"
    log_uniform: str = "log-uniform"


class SklearnTransformerParamModel(BaseModel):
    """
    This represents the meta information needed for a scikit-learn transformer parameter
    """
    name: str
    boundValues: List
    distribution: Optional[DistributionEnum] = Field(None)
    paramType: Literal["numeric", "categorical"] = Field("numeric")

    def to_param(self):
        """
        This converts the meta information into a partial search space
        :return:
        """
        if self.paramType == "categorical":
            return self.boundValues
        else:
            d = self.distribution

            if d is None:
                d = DistributionEnum.uniform

            b = self.boundValues

            return (
                *b,
                d
            )


class SklearnTransformerModel(BaseModel):
    """
    This represents the meta information needed for a scikit-learn transformer
    """

    name: str
    params: Optional[List[Union[SklearnTransformerParamModel]]] = Field(
        None)

    def to_model(self):
        """
        This performs the import of the scikit-learn transformer
        :return:
        """
        if "sklearn" in self.name:
            full_path = self.name
        else:
            full_path = f"sklearn.{self.name}"

        module_path = ".".join(full_path.split(".")[:-1])
        class_path = full_path.split(".")[-1]

        sklearn_module = importlib.import_module(module_path)

        model = getattr(sklearn_module, class_path)()

        return model

    def get_parameter_space(self, prefix: Optional[str] = None):
        """
        This gets the parameter space of the transformer
        :return:
        """

        param_space = {}

        if self.params is None:
            return param_space

        for param in self.params:
            if prefix is None:
                param_space[param.name] = param.to_param()
            else:
                param_space[f"{prefix}{param.name}"] = param.to_param()
        return param_space


class FeaturePodModel(BaseModel):
    """
    This is represents the pod of features and the transformations that need to be applied.
    """
    name: str
    pipeline: List[SklearnTransformerModel]
    features: Optional[List[str]] = None

    def to_sklearn_pipeline(self) -> _ColumnTransformerInput:
        """
        This creates the sklearn pipeline for the features in the pod
        :return:
        """
        steps = []

        for i, transformer_model in enumerate(self.pipeline):
            step = transformer_model.to_model()

            steps.append(
                (f'{self.name}_{i}', step)
            )

        return _ColumnTransformerInput(
            name=self.name,
            sk_obj=Pipeline(
                steps
            ),
            features=self.features
        )

    def to_param_search_space(self, prefix: str) -> Dict:
        """
        This creates the full parameter space for the pod
        :param prefix:
        :return:
        """
        res_params = dict()

        for i, transformer_model in enumerate(self.pipeline):

            if transformer_model.params is None:
                model_param = {}
            else:
                model_param = transformer_model.get_parameter_space()

            res_params = {
                **res_params,
                **dict(
                    (f"{prefix}{self.name}_{i}__{key}", value) for (key, value) in model_param.items()
                )
            }

        return res_params


class MLPipelineStateModel(BaseModel):
    """
    This represents the full pipeline state.

    Here, we should have a scikit-learn model (i.e. Ridge, LogisticRegression) as the model
    parameter, the scoring metric that is supported in scikit-learn, the list of preprocessing
    steps across all of the features, the post process of the resulting features from the application
    of the preprocessing steps, and optionally a transformer model that will convert your target variable
    to the proper state of choice.
    """

    model: SklearnTransformerModel

    scoring: str

    preprocess: Optional[List[FeaturePodModel]] = Field(None)

    postprocess: Optional[FeaturePodModel] = Field(None)

    targetTransformer: Optional[SklearnTransformerModel] = Field(None)

    def to_bayes_opt(self) -> BayesSearchCV:
        """
        This creates the bayesian search CV object with the preprocessing, postprocessing, model and
        target transformer.

        :return:
        """

        if self.preprocess is None:
            steps = []
        else:
            steps = [
                (
                    "preprocess", ColumnTransformer(
                        [pod.to_sklearn_pipeline().to_raw() for pod in self.preprocess]
                    )
                )
            ]

        if self.postprocess is None:
            pass
        else:
            steps.append(
                (
                    "postprocess", self.postprocess.to_sklearn_pipeline().sk_obj
                )
            )

        steps.append(
            (
                "model", self.model.to_model()
            )
        )

        search_params = dict()

        if self.targetTransformer is None:
            base_model = Pipeline(steps)

            if self.preprocess is None:
                pass
            else:
                for x in self.preprocess:
                    search_params = {**search_params, **x.to_param_search_space("preprocess__")}

            if self.postprocess is None:
                pass
            else:
                search_params = {**search_params, **self.postprocess.to_param_search_space("postprocess__")}

            search_params = {**search_params, **self.model.get_parameter_space("model__")}

        else:
            base_model = TransformedTargetRegressor(
                regressor=Pipeline(steps),
                transformer=self.targetTransformer.to_model()
            )

            if self.preprocess is None:
                pass
            else:
                for x in self.preprocess:
                    search_params = {**search_params, **x.to_param_search_space("regressor__preprocess__")}

            if self.postprocess is None:
                pass
            else:
                search_params = {**search_params, **self.postprocess.to_param_search_space("regressor__postprocess__")}

            search_params = {**search_params, **self.model.get_parameter_space("regressor__model__")}

        return BayesSearchCV(
            base_model,
            search_spaces=search_params,
            cv=5,
            scoring=self.scoring
        )
