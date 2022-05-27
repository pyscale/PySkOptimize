import unittest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from skopt.space import Categorical

from pyskoptimize.params import DefaultBooleanParamModel, CategoricalParamModel
from pyskoptimize.steps import SklearnTransformerModel, PostProcessingFeaturePodModel


class TestSklearnTransformModel(unittest.TestCase):
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

    def test_to_model(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
        )

        assert isinstance(model.to_model(), LinearRegression)

    def test_get_parameter_space_no_prefix(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            params=self.params
        )

        param_space = model.get_parameter_space()

        assert "include_bias" in param_space.keys()
        assert isinstance(param_space["include_bias"], Categorical)

    def test_get_parameter_space_prefix(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            params=self.params
        )

        param_space = model.get_parameter_space(prefix="model__")

        assert "model__include_bias" in param_space.keys()
        assert isinstance(param_space["model__include_bias"], Categorical)

    def test_get_default_parameter_space_no_prefix(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            default_params=self.default_params
        )

        param_space = model.get_default_parameter_space()

        assert "include_bias" in param_space.keys()
        assert param_space["include_bias"]

    def test_get_default_parameter_space_prefix(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            default_params=self.default_params
        )

        param_space = model.get_default_parameter_space(prefix="model__")

        assert "model__include_bias" in param_space.keys()
        assert param_space["model__include_bias"]


class TestPostProcessingFeaturePodModel(unittest.TestCase):
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

    def test_to_sklearn_pipeline(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
        )

        pod_model = PostProcessingFeaturePodModel(
            pipeline=[model]
        )

        sk_pipeline = pod_model.to_sklearn_pipeline()

        assert isinstance(sk_pipeline, Pipeline)
        assert isinstance(sk_pipeline.steps[0][-1], LinearRegression)

    def test_get_parameter_space(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            params=self.params
        )

        pod_model = PostProcessingFeaturePodModel(
            pipeline=[model]
        )

        param_space = pod_model.get_parameter_space(prefix="post")
        print(param_space)
        assert "post__0__include_bias" in param_space.keys()
        assert isinstance(param_space["post__0__include_bias"], Categorical)

    def test_get_parameter_space_no_params(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
        )

        pod_model = PostProcessingFeaturePodModel(
            pipeline=[model]
        )

        param_space = pod_model.get_parameter_space(prefix="post")

        assert len(list(param_space.keys())) == 0

    def test_get_default_parameter_space(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            default_params=self.default_params
        )

        pod_model = PostProcessingFeaturePodModel(
            pipeline=[model]
        )

        param_space = pod_model.get_default_parameter_space(prefix="post")

        assert "post__0__include_bias" in param_space.keys()
        assert param_space["post__0__include_bias"]

    def test_get_default_parameter_space_no_params(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
        )

        pod_model = PostProcessingFeaturePodModel(
            pipeline=[model]
        )

        param_space = pod_model.get_default_parameter_space(prefix="post")

        assert len(list(param_space.keys())) == 0
