from abc import abstractmethod, ABC
from typing import Dict, List

from sklearn.compose import ColumnTransformer
from pydantic import BaseModel

from .steps import SklearnTransformerModel, PreprocessingFeaturePodModel, PostProcessingFeaturePodModel


class HasEstimator(BaseModel):
    """
    The estimator trait

    :var model: The sklearn transformer configurations for the model
    """

    model: SklearnTransformerModel

    @property
    def estimator(self):
        """
        This generates the base estimator for the machine learning task

        :return: the base estimator
        """
        return self.model.to_model()

    @property
    def estimator_param_space(self) -> Dict:
        """
        This generates the parameter space for the Bayesian search

        :return: The search space
        """
        search_params = self.model.get_parameter_space("model__")

        return search_params


class HasFeaturePreprocessing(BaseModel):
    """
    The feature preprocessing trait

    :var preprocess: The list of preprocessing applications on different features
    """

    preprocess: List[PreprocessingFeaturePodModel]

    @property
    def preprocess_pipeline(self):
        """
        This generates the base estimator for the machine learning task

        :return: the base estimator
        """
        base_model = ColumnTransformer(
            [(pod.name, pod.to_sklearn_pipeline(), pod.features) for pod in self.preprocess]
        )

        return base_model

    @property
    def preprocess_pipeline_param_space(self):
        """
        This generates the parameter space for the Bayesian search

        :return: The search space
        """

        search_params = dict()

        for x in self.preprocess:
            search_params = {**search_params, **x.get_parameter_space("preprocess")}

        return search_params


class HasFeaturePostProcessing(BaseModel):
    """
    The feature post processing trait

    :var postprocess: The list of postprocessing steps to apply onto the feature union
    """

    postProcess: PostProcessingFeaturePodModel

    @property
    def post_process_pipeline(self):
        """
        This generates the base estimator for the machine learning task

        :return: the base estimator
        """
        base_model = self.postProcess.to_sklearn_pipeline()

        return base_model

    @property
    def post_process_pipeline_param_space(self):
        """
        This generates the parameter space for the Bayesian search

        :return: The search space
        """

        search_params = self.postProcess.get_parameter_space("postprocess")

        return search_params


class IsMLPipeline(ABC):
    """
    The base ML Pipeline to standardize which properties to expect
    """

    @property
    @abstractmethod
    def pipeline(self):
        pass

    @property
    @abstractmethod
    def parameter_space(self):
        pass

