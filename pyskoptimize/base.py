import importlib
from abc import ABCMeta, abstractmethod

from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from skopt.searchcv import BayesSearchCV
from skopt.space import Integer, Categorical, Real

Numeric = Union[float, int]


@dataclass(frozen=True)
class _ColumnTransformerInput:
    """
    This is a private class to handle raw input

    :var name: The name of the column transformation
    :var sk_obj: The pipeline of transformations
    :var features: The optional list of features
    """
    name: str
    sk_obj: Pipeline
    features: List[str]

    def to_raw(self) -> Tuple:
        """
        This returns the raw values needed for the ColumnTransformer or the Pipeline
        :return: The raw values
        """
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


class BaseParamModel(BaseModel, metaclass=ABCMeta):
    """
    The base abstract class for all scikit-learn compatible parameters

    :var name: The name of the parameter
    """
    name: str

    @abstractmethod
    def to_param(self):
        """
        The abstract class to get the parameter space with the current distribution
        :return: The skopt parameter
        """


class CategoricalParamModel(BaseParamModel):
    """
    The class to handle categorical parameter values

    :var name: The name of the parameter
    :var categories: The list of categories we are going to use
    """
    categories: List

    def to_param(self):
        """
        This converts the param to what skopt needs

        :return: The categorical
        """
        return Categorical(self.categories)


class UniformlyDistributedIntegerParamModel(BaseParamModel):
    """
    This is for the uniform integer distribution
    """
    lowInt: int
    highInt: int

    def to_param(self):
        """
        This converts the param to what skopt needs

        :return: The integer parameter
        """
        return Integer(self.lowInt, self.highInt)


class NumericParamModel(BaseParamModel, metaclass=ABCMeta):
    """
    An abstract base class for all purely numeric parameters

    :var name: The name of the parameter
    :var log_scale: A boolean if we are using the log scale
    """
    log_scale: bool = Field(False)


class UniformlyDistributedParamModel(NumericParamModel):
    """
    The class for uniformly (or log-uniformly) distributed parameters

    :var name: The name of the parameter
    :var low: The lowest value
    :var high: The highest value
    :var log_scale: A boolean if we are using the log scale
    """
    low: Numeric
    high: Numeric

    def to_param(self):
        """
        This converts the param to what skopt needs

        :return: The skopt parameter
        """

        if self.log_scale:
            d = "log-uniform"
        else:
            d = "uniform"

        return Real(
            self.low,
            self.high,
            prior=d
        )


class NormallyDistributedParamModel(NumericParamModel):
    """
    The class for normally (or log-normally) distributed parameters

    :var name: The name of the parameter
    :var mu: The mean
    :var sigma: The variance
    :var log_scale: A boolean if we are using the log scale
    """
    mu: Numeric
    sigma: Numeric

    def to_param(self) -> Tuple:
        """
        This converts the param to what skopt needs

        :return: The skopt parameter
        """

        if self.log_scale:
            d = "log-normal"
        else:
            d = "normal"

        return (
            self.mu,
            self.sigma,
            d
        )


class SklearnTransformerModel(BaseModel):
    """
    This represents the meta information needed for a scikit-learn transformer

    :var name: The name of the transformer
    :var params: The parameters to perform the bayesian optimization on
    """

    name: str
    params: Optional[
        List[
            Union[
                NormallyDistributedParamModel,
                UniformlyDistributedParamModel,
                CategoricalParamModel,
                UniformlyDistributedIntegerParamModel
            ]
        ]
    ] = Field(None)

    def to_model(self):
        """
        This performs the import of the scikit-learn transformer

        :return: The sklearn object
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

        :return: The parameter search
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

    :var name: The name of the feature pod
    :var pipeline: The list of transformations to apply onto features
    :var features: The optional list of features
    """

    pipeline: List[SklearnTransformerModel]
    name: Optional[str] = None
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
                (f'{i}', step)
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
        if self.name is None:
            name = ""
        else:
            name = self.name

        for i, transformer_model in enumerate(self.pipeline):

            if transformer_model.params is None:
                model_param = {}
            else:
                model_param = transformer_model.get_parameter_space()

            res_params = {
                **res_params,
                **dict(
                    (f"{prefix}{name}__{i}__{key}", value) for (key, value) in model_param.items()
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

    :var model: The sklearn transformer configurations for the model
    :var scoring: The scoring metric
    :var preprocess: The list of preprocessing applications on different features
    :var postprocess: The list of postprocessing steps to apply onto the feature union
    :var targetTransformer: The sklearn transformer to apply onto the target label before training the model
    :var cv: The cross validation number
    """

    model: SklearnTransformerModel

    scoring: str

    preprocess: Optional[List[FeaturePodModel]] = Field(None)

    postprocess: Optional[FeaturePodModel] = Field(None)

    targetTransformer: Optional[SklearnTransformerModel] = Field(None)

    cv: int = Field(5)

    def to_sk_obj(self):
        """
        This generates the base estimator for the machine learning task

        :return: the base estimator
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

        if self.targetTransformer is None:
            base_model = Pipeline(steps)
        else:
            base_model = TransformedTargetRegressor(
                regressor=Pipeline(steps),
                transformer=self.targetTransformer.to_model()
            )

        return base_model

    def to_param_space(self):
        """
        This generates the parameter space for the Bayesian search

        :return: The search space
        """
        search_params = {}

        if self.targetTransformer is None:
            if self.preprocess is None:
                pass
            else:
                for x in self.preprocess:
                    search_params = {**search_params, **x.to_param_search_space("preprocess__")}

            if self.postprocess is None:
                pass
            else:
                search_params = {**search_params, **self.postprocess.to_param_search_space("postprocess")}

            search_params = {**search_params, **self.model.get_parameter_space("model__")}
        else:
            if self.preprocess is None:
                pass
            else:
                for x in self.preprocess:
                    search_params = {**search_params, **x.to_param_search_space("regressor__preprocess__")}

            if self.postprocess is None:
                pass
            else:
                search_params = {**search_params, **self.postprocess.to_param_search_space("regressor__postprocess")}

            search_params = {**search_params, **self.model.get_parameter_space("regressor__model__")}

        return search_params

    def to_bayes_opt(self, verbose: int = 0) -> BayesSearchCV:
        """
        This creates the bayesian search CV object with the preprocessing, postprocessing, model and
        target transformer.

        :return: The bayesian search method with the base estimator and search space
        """

        base_estimator = self.to_sk_obj()
        search_parameter_space = self.to_param_space()

        return BayesSearchCV(
            base_estimator,
            search_spaces=search_parameter_space,
            cv=5,
            scoring=self.scoring,
            verbose=verbose
        )
