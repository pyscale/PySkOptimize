from unittest import TestCase

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skopt.space import Categorical

from pyskoptimize.traits import HasFeaturePreprocessing, HasFeaturePostProcessing
from pyskoptimize.params import DefaultBooleanParamModel, CategoricalParamModel
from pyskoptimize.steps import SklearnTransformerModel, PostProcessingFeaturePodModel, PreprocessingFeaturePodModel


class TestHasFeaturePreprocessing(TestCase):
    """

    """

    def setUp(self) -> None:
            self.name = "sklearn.linear_model.LinearRegression"
            self.params = [
                CategoricalParamModel(
                    name="include_bias",
                    categories=[True, False]
                )
            ]
            self.default_params = [
                DefaultBooleanParamModel(
                    name="include_bias",
                    valueBool=True
                )
            ]

    def test_property_preprocess_pipeline(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
        )

        pod_model = PreprocessingFeaturePodModel(
            name="features",
            pipeline=[model],
            features=["dummy"]
        )

        pre_pipeline = HasFeaturePreprocessing(preprocess=[pod_model]).preprocess_pipeline

        assert isinstance(pre_pipeline, ColumnTransformer)

    def test_preprocess_pipeline_param_space(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            params=self.params
        )

        pod_model = PreprocessingFeaturePodModel(
            name="features",
            pipeline=[model],
            features=["dummy"]
        )

        pre_pipeline_space = HasFeaturePreprocessing(preprocess=[pod_model]).preprocess_pipeline_param_space

        assert isinstance(pre_pipeline_space["preprocess__features__0__include_bias"], Categorical)


class TestHasFeaturePostProcessing(TestCase):
    """

    """

    def setUp(self) -> None:
            self.name = "sklearn.linear_model.LinearRegression"
            self.params = [
                CategoricalParamModel(
                    name="include_bias",
                    categories=[True, False]
                )
            ]
            self.default_params = [
                DefaultBooleanParamModel(
                    name="include_bias",
                    valueBool=True
                )
            ]

    def test_property_preprocess_pipeline(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
        )

        pod_model = PostProcessingFeaturePodModel(
            pipeline=[model],
        )

        pre_pipeline = HasFeaturePostProcessing(postProcess=pod_model).post_process_pipeline

        assert isinstance(pre_pipeline, Pipeline)

    def test_preprocess_pipeline_param_space(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            params=self.params
        )

        pod_model = PostProcessingFeaturePodModel(
            pipeline=[model],
        )

        pre_pipeline_space = HasFeaturePostProcessing(postProcess=pod_model).post_process_pipeline_param_space

        assert isinstance(pre_pipeline_space["postprocess__0__include_bias"], Categorical)
