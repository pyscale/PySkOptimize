import unittest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from skopt.space import Categorical

from pyskoptimize.params import DefaultBooleanParamModel, CategoricalParamModel
from pyskoptimize.steps import SklearnTransformerModel, PostProcessingFeaturePodModel, PreprocessingFeaturePodModel


class TestSklearnTransformModel(unittest.TestCase):
    """

    """

    def setUp(self) -> None:

        self.name = "sklearn.linear_model.LinearRegression"
        self.params = [
            CategoricalParamModel(
                name="fit_intercept",
                categories=[True, False]
            )
        ]
        self.default_params = [
            DefaultBooleanParamModel(
                name="fit_intercept",
                valueBool=False
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

        assert "fit_intercept" in param_space.keys()
        assert isinstance(param_space["fit_intercept"], Categorical)

    def test_get_parameter_space_prefix(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            params=self.params
        )

        param_space = model.get_parameter_space(prefix="model__")

        assert "model__fit_intercept" in param_space.keys()
        assert isinstance(param_space["model__fit_intercept"], Categorical)

    def test_get_default_parameter_space(self):
        """

        :return:
        """
        model = SklearnTransformerModel(
            name=self.name,
            default_params=self.default_params
        )

        param_space = model.get_default_parameter_space()

        assert "fit_intercept" in list(param_space.keys())
        assert param_space["fit_intercept"] is False


class TestPostProcessingFeaturePodModel(unittest.TestCase):
    """

    """

    def setUp(self) -> None:

        self.name = "sklearn.linear_model.LinearRegression"
        self.params = [
            CategoricalParamModel(
                name="fit_intercept",
                categories=[True, False]
            )
        ]
        self.default_params = [
            DefaultBooleanParamModel(
                name="fit_intercept",
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

        param_space = pod_model.get_parameter_space(name="post")

        assert "post__0__fit_intercept" in param_space.keys()
        assert isinstance(param_space["post__0__fit_intercept"], Categorical)

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

        param_space = pod_model.get_parameter_space(name="post")

        assert len(list(param_space.keys())) == 0


class TestPreProcessingFeaturePodModel(unittest.TestCase):
    """

    """

    def setUp(self) -> None:

        self.name = "sklearn.linear_model.LinearRegression"
        self.params = [
            CategoricalParamModel(
                name="fit_intercept",
                categories=[True, False]
            )
        ]
        self.default_params = [
            DefaultBooleanParamModel(
                name="fit_intercept",
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

        pod_model = PreprocessingFeaturePodModel(
            name="features",
            pipeline=[model],
            features=["dummy"]
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

        pod_model = PreprocessingFeaturePodModel(
            name="features",
            pipeline=[model],
            features=["dummy"]
        )

        param_space = pod_model.get_parameter_space(prefix="post")

        assert "post__features__0__fit_intercept" in list(param_space.keys())
        assert isinstance(param_space["post__features__0__fit_intercept"], Categorical)

    def test_get_parameter_space_no_params(self):
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

        param_space = pod_model.get_parameter_space(prefix="post")

        assert len(list(param_space.keys())) == 0
