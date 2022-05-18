import importlib

from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from skopt.searchcv import BayesSearchCV

from pyskoptimize.params import UniformlyDistributedIntegerParamModel, \
    CategoricalParamModel, UniformlyDistributedParamModel, NormallyDistributedParamModel, \
    DefaultFloatParamModel, DefaultIterableParamModel, DefaultBooleanParamModel, DefaultStringParamModel, \
    DefaultIntegerParamModel


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

    default_params: Optional[
        List[
            Union[
                DefaultFloatParamModel,
                DefaultIterableParamModel,
                DefaultBooleanParamModel,
                DefaultStringParamModel,
                DefaultIntegerParamModel
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
    name: str
    features: List[str]

    def to_sklearn_pipeline(self) -> Pipeline:
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

        return Pipeline(
                steps
            )

    def to_param_search_space(self, prefix: str) -> Dict:
        """
        This creates the full parameter space for the pod

        :param prefix:

        :return:
        """
        res_params = dict()

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

    postprocess: Optional[List[SklearnTransformerModel]] = Field(None)

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
                        [(pod.name, pod.to_sklearn_pipeline(), pod.features) for pod in self.preprocess]
                    )
                )
            ]

        if self.postprocess is None:
            pass
        else:
            _steps = []

            for i, transformer_model in enumerate(self.postprocess):
                step = transformer_model.to_model()

                _steps.append(
                    (f'{i}', step)
                )

            steps.append(
                (
                    "postprocess", Pipeline(_steps)
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
                res_params = dict()

                for i, transformer_model in enumerate(self.postprocess):

                    if transformer_model.params is None:
                        model_param = {}
                    else:
                        model_param = transformer_model.get_parameter_space()

                    res_params = {
                        **res_params,
                        **dict(
                            (f"postprocess__{i}__{key}", value) for (key, value) in model_param.items()
                        )
                    }

                search_params = {**search_params, **res_params}

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
                res_params = dict()

                for i, transformer_model in enumerate(self.postprocess):

                    if transformer_model.params is None:
                        model_param = {}
                    else:
                        model_param = transformer_model.get_parameter_space()

                    res_params = {
                        **res_params,
                        **dict(
                            (f"regressor__postprocess__{i}__{key}", value) for (key, value) in model_param.items()
                        )
                    }

                search_params = {**search_params, **res_params}

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
