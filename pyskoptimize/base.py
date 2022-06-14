from typing import Union, Dict

from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from skopt.space import Categorical, Integer, Real
from skopt.searchcv import BayesSearchCV

from .steps import SklearnTransformerModel
from .traits import HasEstimator, HasFeaturePreprocessing, HasFeaturePostProcessing, IsMLPipeline

SkOptHyperParameters = Union[Categorical, Integer, Real]


class MLEstimator(HasEstimator, IsMLPipeline):
    """
    This is the ML Estimator, with no feature engineering
    """

    @property
    def pipeline(self) -> Pipeline:
        """
        The ML Pipeline
        :return:
        """
        estimator = self.estimator

        return Pipeline(
            steps=[
                ("model", estimator)
            ]
        )

    @property
    def parameter_space(self) -> Dict[str, SkOptHyperParameters]:
        """
        The tuning parameter space

        :return:
        """
        return self.estimator_param_space


class MLEstimatorWithFeaturePreprocess(HasEstimator, HasFeaturePreprocessing, IsMLPipeline):
    """
    This is the ML Estimator, with groupings of feature engineering
    """

    @property
    def pipeline(self) -> Pipeline:
        """
        The ML Pipeline

        :return:
        """
        estimator = self.estimator

        feature_pipeline = self.preprocess_pipeline

        return Pipeline(
            steps=[
                ("preprocess", feature_pipeline),
                ("model", estimator)
            ]
        )

    @property
    def parameter_space(self) -> Dict[str, SkOptHyperParameters]:
        """
        The tuning parameter space

        :return:
        """
        return {**self.estimator_param_space, **self.preprocess_pipeline_param_space}


class MLEstimatorWithFeaturePostProcess(HasEstimator, HasFeaturePostProcessing, IsMLPipeline):
    """
    This is the ML Estimator, with feature engineering on assuming processed feature

    """

    @property
    def pipeline(self) -> Pipeline:
        """
        The ML Pipeline

        :return:
        """
        return Pipeline(
            steps=[
                ("postprocess", self.post_process_pipeline),
                ("model", self.estimator)
            ]
        )

    @property
    def parameter_space(self) -> Dict[str, SkOptHyperParameters]:
        """
        The tuning parameter space

        :return:
        """
        return {**self.estimator_param_space, **self.post_process_pipeline_param_space}


class MLEstimatorWithFeaturePrePostProcess(HasEstimator, HasFeaturePreprocessing, HasFeaturePostProcessing, IsMLPipeline):
    """
    This is the ML Estimator, with feature engineering, allowing for grouping of feature engineering and a final
    aggregate of feature engineering

    """

    @property
    def pipeline(self) -> Pipeline:
        """
        The ML Pipeline

        :return:
        """
        return Pipeline(
            steps=[
                ("preprocess", self.preprocess_pipeline),
                ("postprocess", self.post_process_pipeline),
                ("model", self.estimator),
            ]
        )

    @property
    def parameter_space(self) -> Dict[str, SkOptHyperParameters]:
        """
        The tuning parameter space

        :return:
        """
        return {
            **self.estimator_param_space,
            **self.preprocess_pipeline_param_space,
            **self.post_process_pipeline_param_space
        }


class TargetTransformationMLPipeline(BaseModel, IsMLPipeline):
    """
    This is the target transformation pipeline, which requires a base estimator

    """
    baseEstimator: Union[
        MLEstimator,
        MLEstimatorWithFeaturePreprocess,
        MLEstimatorWithFeaturePostProcess,
        MLEstimatorWithFeaturePrePostProcess
    ]

    targetTransformer: SklearnTransformerModel

    @property
    def pipeline(self) -> TransformedTargetRegressor:
        """
        The ML pipeline with the target transformation

        :return:
        """
        target_transformer = self.targetTransformer.to_model()

        base_model = TransformedTargetRegressor(
            regressor=self.baseEstimator.pipeline,
            transformer=target_transformer
        )

        return base_model

    @staticmethod
    def _parameter_space(param_space) -> Dict:
        """
        A private function to help change the namespace of the parameter space for the regressor

        :param param_space: The initial parameter space
        :return:
        """
        keys = list(param_space.keys())

        search_params = {f"regressor__{key}": param_space[key] for key in keys}

        return search_params

    @property
    def parameter_space(self) -> Dict[str, SkOptHyperParameters]:
        """
        The tuning parameter space

        :return:
        """

        return self._parameter_space(self.baseEstimator.parameter_space)


class MLOptimizer(BaseModel):
    """
    This represents the full pipeline state.

    Here, we should have a scikit-learn model (i.e. Ridge, LogisticRegression) as the model
    parameter, the scoring metric that is supported in scikit-learn, the list of preprocessing
    steps across all of the features, the post process of the resulting features from the application
    of the preprocessing steps, and optionally a transformer model that will convert your target variable
    to the proper state of choice.

    :var mlPipeline: The pipeline we want to optimize
    :var scoring: The scoring metric
    :var cv: The cross validation number
    """

    mlPipeline: Union[
        TargetTransformationMLPipeline,
        MLEstimatorWithFeaturePrePostProcess,
        MLEstimatorWithFeaturePostProcess,
        MLEstimatorWithFeaturePreprocess,
        MLEstimator
    ]

    scoring: str

    cv: int = Field(5)

    def to_bayes_opt(self, verbose: int = 0, n_iter: int = 50) -> BayesSearchCV:
        """
        This creates the bayesian search CV object with the preprocessing, postprocessing, model and
        target transformer.

        :return: The bayesian search method with the base estimator and search space
        """

        ml_pipeline = self.mlPipeline.pipeline

        search_parameter_space = self.mlPipeline.parameter_space

        assert 0 < len(list(search_parameter_space.keys())), """
            There are no search parameters.  If you do not need to tune your parameters,
            please just use the create the pipeline yourself. 
        """

        return BayesSearchCV(
            ml_pipeline,
            search_spaces=search_parameter_space,
            cv=5,
            scoring=self.scoring,
            verbose=verbose,
            n_iter=n_iter
        )
