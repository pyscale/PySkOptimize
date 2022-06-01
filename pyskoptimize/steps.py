from abc import ABCMeta
from typing import Optional, List, Union, Dict

from pydantic import BaseModel, PyObject, Field
from sklearn.pipeline import Pipeline

from pyskoptimize.params import NormallyDistributedParamModel, UniformlyDistributedParamModel, CategoricalParamModel, \
    UniformlyDistributedIntegerParamModel, DefaultFloatParamModel, DefaultCollectionParamModel, \
    DefaultBooleanParamModel, DefaultStringParamModel, DefaultIntegerParamModel, \
    BaseParamModel, HasParameterSpace


class SklearnTransformerModel(BaseModel, HasParameterSpace):
    """
    This represents the meta information needed for a scikit-learn transformer

    :var name: The name of the transformer
    :var params: The parameters to perform the bayesian optimization on
    :var default_params: The default parameters
    """

    name: PyObject
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
                DefaultCollectionParamModel,
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

        model = self.name()

        default_params = self.get_default_parameter_space()

        if len(list(default_params.keys())) == 0:
            pass
        else:
            model.set_params(**default_params)

        return model

    @staticmethod
    def _get_space(prefix: Optional[str], params: List[BaseParamModel]):
        """

        :param prefix:
        :return:
        """
        param_space = dict()

        if params is None:
            return param_space

        for param in params:
            if prefix is None:
                param_space[param.name] = param.to_param()
            else:
                param_space[f"{prefix}{param.name}"] = param.to_param()
        return param_space

    def get_parameter_space(self, prefix: Optional[str] = None):
        """
        This gets the parameter space of the transformer

        :return: The parameter search
        """

        return self._get_space(prefix=prefix, params=self.params)

    def get_default_parameter_space(self) -> Dict:
        """
        This gets the parameter space of the transformer

        :return: The parameter search
        """

        return self._get_space(prefix=None, params=self.default_params)


class FeatureProcessingModel(BaseModel, HasParameterSpace, metaclass=ABCMeta):
    """

    """
    pipeline: List[SklearnTransformerModel]

    def to_sklearn_pipeline(self) -> Pipeline:
        """
        This creates the sklearn pipeline for the features in the pod

        :return: A sklearn Pipeline that represents the feature pod
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

    def _get_parameter_space(self, prefix: str) -> Dict:
        """
        The private method for creating the parameter search space
        :param prefix: The name to prefix the parameters
        :return: A dictionray of the parameter search space
        """
        res_params = dict()

        for i, transformer_model in enumerate(self.pipeline):

            if transformer_model.params is None:
                model_param = {}
            else:
                model_param = transformer_model.get_parameter_space(prefix=f"{prefix}__{i}__")

            res_params = {
                **res_params,
                **model_param
            }

        return res_params


class PostProcessingFeaturePodModel(FeatureProcessingModel):
    """
    This is represents the pod of features and the transformations that need to be applied.

    """

    def get_parameter_space(self, name: str) -> Dict:
        """

        :param name:
        :return:
        """
        return self._get_parameter_space(name)


class PreprocessingFeaturePodModel(FeatureProcessingModel):
    """
    This is represents the pod of features and the transformations that need to be applied.

    :var name: The name of the feature pod
    :var pipeline: The list of transformations to apply onto features
    :var features: The optional list of features
    """

    name: str
    features: List[str]

    def get_parameter_space(self, prefix: str) -> Dict:
        """

        :param prefix:
        :return:
        """
        key = f"{prefix}__{self.name}"
        return self._get_parameter_space(key)
