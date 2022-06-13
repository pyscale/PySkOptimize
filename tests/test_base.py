from unittest import TestCase

from sklearn.pipeline import Pipeline
from skopt.space import Real, Integer

from pyskoptimize.base import MLEstimator, MLEstimatorWithFeaturePreprocess, \
    MLEstimatorWithFeaturePostProcess, MLEstimatorWithFeaturePrePostProcess


class TestMLEstimator(TestCase):
    """

    """

    def setUp(self) -> None:
        self.params = {
            "model": {
                "name": "sklearn.linear_model.Ridge",
                "params": [
                    {
                        "name": "alpha",
                        "low": 1e-16,
                        "high": 1e16,
                        "log_scale": True
                    }
                ],
                "default_params": [
                    {
                        "name": "fit_intercept",
                        "valueBool": False
                    }
                ]
            }
        }

    def test_pipeline(self):
        """

        :return:
        """

        estimator: MLEstimator = MLEstimator.parse_obj(self.params)

        assert isinstance(estimator.pipeline, Pipeline)
        assert len(estimator.pipeline.steps) == 1

    def test_tuning_param_space(self):
        """

        :return:
        """
        estimator: MLEstimator = MLEstimator.parse_obj(self.params)

        assert isinstance(estimator.parameter_space["model__alpha"], Real)


class TestMLEstimatorWithFeaturePreprocess(TestCase):
    """

    """

    def setUp(self) -> None:
        self.params = {
            "model": {
                "name": "sklearn.linear_model.Ridge",
                "params": [
                    {
                        "name": "alpha",
                        "low": 1e-16,
                        "high": 1e16,
                        "log_scale": True
                    }
                ],
                "default_params": [
                    {
                        "name": "fit_intercept",
                        "valueBool": False
                    }
                ]
            },
            "preprocess": [
                {
                    "name": "features",
                    "pipeline": [
                        {
                            "name": "sklearn.feature_selection.GenericUnivariateSelect",
                            "params": [
                                {
                                    "name": "param",
                                    "low": 1e-16,
                                    "high": 1.
                                }
                            ],
                            "default_params": [
                                {
                                    "name": "mode",
                                    "valueStr": "fwe"
                                }
                            ]
                        }
                    ],
                    "features": ["dummy"]
                }
            ]
        }

    def test_pipeline(self):
        """

        :return:
        """

        estimator: MLEstimatorWithFeaturePreprocess = MLEstimatorWithFeaturePreprocess.parse_obj(self.params)

        assert isinstance(estimator.pipeline, Pipeline)
        assert len(estimator.pipeline.steps) == 2

    def test_tuning_param_space(self):
        """

        :return:
        """
        estimator: MLEstimatorWithFeaturePreprocess = MLEstimatorWithFeaturePreprocess.parse_obj(self.params)

        assert isinstance(estimator.parameter_space["model__alpha"], Real)
        assert isinstance(estimator.parameter_space["preprocess__features__0__param"], Real)


class TestMLEstimatorWithFeaturePostProcess(TestCase):
    """

    """

    def setUp(self) -> None:
        self.params = {
            "model": {
                "name": "sklearn.linear_model.Ridge",
                "params": [
                    {
                        "name": "alpha",
                        "low": 1e-16,
                        "high": 1e16,
                        "log_scale": True
                    }
                ],
                "default_params": [
                    {
                        "name": "fit_intercept",
                        "valueBool": False
                    }
                ]
            },
            "postProcess": {
                "pipeline": [
                    {
                        "name": "sklearn.feature_selection.GenericUnivariateSelect",
                        "params": [
                            {
                                "name": "param",
                                "low": 1e-16,
                                "high": 1.
                            }
                        ],
                        "default_params": [
                            {
                                "name": "mode",
                                "valueStr": "fwe"
                            }
                        ]
                    }
                ],
            }
        }

    def test_pipeline(self):
        """

        :return:
        """

        estimator: MLEstimatorWithFeaturePostProcess = MLEstimatorWithFeaturePostProcess.parse_obj(self.params)

        assert isinstance(estimator.pipeline, Pipeline)
        assert len(estimator.pipeline.steps) == 2

    def test_tuning_param_space(self):
        """

        :return:
        """
        estimator: MLEstimatorWithFeaturePostProcess = MLEstimatorWithFeaturePostProcess.parse_obj(self.params)

        assert isinstance(estimator.parameter_space["model__alpha"], Real)
        assert isinstance(estimator.parameter_space["postprocess__0__param"], Real)


class TestMLEstimatorWithFeaturePrePostProcess(TestCase):
    """

    """

    def setUp(self) -> None:
        self.params = {
            "model": {
                "name": "sklearn.linear_model.Ridge",
                "params": [
                    {
                        "name": "alpha",
                        "low": 1e-16,
                        "high": 1e16,
                        "log_scale": True
                    }
                ],
                "default_params": [
                    {
                        "name": "fit_intercept",
                        "valueBool": False
                    }
                ]
            },
            "preprocess": [
                {
                    "name": "features",
                    "pipeline": [
                        {
                            "name": "sklearn.decomposition.PCA",
                            "params": [
                                {
                                    "name": "n_components",
                                    "lowInt": 2,
                                    "highInt": 3
                                }
                            ],
                            "default_params": [
                                {
                                    "name": "whiten",
                                    "valueBool": True
                                }
                            ]
                        }
                    ],
                    "features": ["dummy1", "dummy2", "dummy3"]
                }
            ],
            "postProcess": {
                "pipeline": [
                    {
                        "name": "sklearn.feature_selection.GenericUnivariateSelect",
                        "params": [
                            {
                                "name": "param",
                                "low": 1e-16,
                                "high": 1.
                            }
                        ],
                        "default_params": [
                            {
                                "name": "mode",
                                "valueStr": "fwe"
                            }
                        ]
                    }
                ],
            }
        }

    def test_pipeline(self):
        """

        :return:
        """

        estimator: MLEstimatorWithFeaturePrePostProcess = MLEstimatorWithFeaturePrePostProcess.parse_obj(self.params)

        assert isinstance(estimator.pipeline, Pipeline)
        assert len(estimator.pipeline.steps) == 3

    def test_tuning_param_space(self):
        """

        :return:
        """
        estimator: MLEstimatorWithFeaturePrePostProcess = MLEstimatorWithFeaturePrePostProcess.parse_obj(self.params)

        assert isinstance(estimator.parameter_space["model__alpha"], Real)
        assert isinstance(estimator.parameter_space["postprocess__0__param"], Real)
        assert isinstance(estimator.parameter_space["preprocess__features__0__n_components"], Integer)

