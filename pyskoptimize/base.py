from typing import Union

from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from skopt.searchcv import BayesSearchCV

from .steps import SklearnTransformerModel
from .traits import HasEstimator, HasFeaturePreprocessing, HasFeaturePostProcessing, IsMLPipeline


class MLEstimator(HasEstimator, IsMLPipeline):
    """
    This is just a wrapper
    """

    @property
    def pipeline(self):
        """

        :return:
        """
        estimator = self.estimator

        return Pipeline(
            steps=[
                ("model", estimator)
            ]
        )

    @property
    def parameter_space(self):
        """

        :return:
        """
        return self.estimator_param_space

    @property
    def default_parameters(self):
        """

        :return:
        """
        return self.estimator_default_parameters


class MLEstimatorWithFeaturePreprocess(HasEstimator, HasFeaturePreprocessing, IsMLPipeline):
    """

    """

    @property
    def pipeline(self):
        """

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
    def parameter_space(self):
        """

        :return:
        """
        return {**self.estimator_param_space, **self.preprocess_pipeline_param_space}

    @property
    def default_parameters(self):
        """

        :return:
        """
        return {**self.estimator_default_parameters, **self.preprocess_pipeline_default_parameters}


class MLEstimatorWithFeaturePostProcess(HasEstimator, HasFeaturePostProcessing, IsMLPipeline):
    """

    """

    @property
    def pipeline(self):
        """

        :return:
        """
        return Pipeline(
            steps=[
                ("postprocess", self.post_process_pipeline),
                ("model", self.estimator)
            ]
        )

    @property
    def parameter_space(self):
        """

        :return:
        """
        return {**self.estimator_param_space, **self.post_process_pipeline_param_space}

    @property
    def default_parameters(self):
        """

        :return:
        """
        return {**self.estimator_default_parameters, **self.post_process_pipeline_default_parameters}


class MLEstimatorWithFeaturePrePostProcess(HasEstimator, HasFeaturePreprocessing, HasFeaturePostProcessing, IsMLPipeline):
    """

    """

    @property
    def pipeline(self):
        """

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
    def parameter_space(self):
        """

        :return:
        """
        return {**self.estimator_param_space, **self.preprocess_pipeline_param_space, **self.post_process_pipeline}

    @property
    def default_parameters(self):
        """

        :return:
        """
        return {
            **self.estimator_default_parameters,
            **self.preprocess_pipeline_default_parameters,
            **self.post_process_pipeline_default_parameters
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
    def pipeline(self):

        target_transformer = self.targetTransformer.to_model()

        base_model = TransformedTargetRegressor(
            regressor=self.baseEstimator.pipeline,
            transformer=target_transformer
        )

        return base_model

    @staticmethod
    def _parameter_space(param_space):
        """

        :param param_space:
        :return:
        """
        keys = list(param_space.keys())

        search_params = {f"regressor__{key}": param_space[key] for key in keys}

        return search_params

    @property
    def parameter_space(self):
        """

        :return:
        """

        return self._parameter_space(self.baseEstimator.parameter_space)

    @property
    def default_parameters(self):
        """

        :return:
        """
        return self._parameter_space(self.baseEstimator.default_parameters)


class MLOptimizer(BaseModel):
    """
    This represents the full pipeline state.

    Here, we should have a scikit-learn model (i.e. Ridge, LogisticRegression) as the model
    parameter, the scoring metric that is supported in scikit-learn, the list of preprocessing
    steps across all of the features, the post process of the resulting features from the application
    of the preprocessing steps, and optionally a transformer model that will convert your target variable
    to the proper state of choice.

    :var pipeline: The pipeline we want to optimize
    :var scoring: The scoring metric
    :var cv: The cross validation number
    """

    pipeline: Union[
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

        ml_pipeline = self.pipeline.pipeline

        search_parameter_space = self.pipeline.parameter_space

        default_parameter_space = self.pipeline.default_parameters

        if len(list(default_parameter_space.keys())) == 0:
            pass
        else:
            ml_pipeline.set_params(**default_parameter_space)

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
