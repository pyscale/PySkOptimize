from pyskoptimize.base import MLPipelineStateModel

from sklearn.linear_model import Ridge


def test_from_model_to_model(demo_simple_housing: MLPipelineStateModel):
    """
    This just tests if it can properly parse the models

    :return:
    """

    ridge_model = demo_simple_housing.model.to_model()

    assert isinstance(ridge_model, Ridge)


def test_from_model_to_model_params(demo_simple_housing: MLPipelineStateModel):
    """
    This just tests if we can parse the model parameters
    :return:
    """

    assert demo_simple_housing.model.params[0].log_scale

    ridge_model_params = demo_simple_housing.model.get_parameter_space()

    assert isinstance(ridge_model_params, dict)
    assert len(ridge_model_params["alpha"]) == 3
    assert ridge_model_params['alpha'][0] == 1e-16
    assert ridge_model_params['alpha'][1] == 1e16
    assert ridge_model_params['alpha'][2] == "log-uniform"