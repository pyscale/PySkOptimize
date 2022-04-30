from api.models import MLPipelineStateModel

from sklearn.linear_model import Ridge


def test_from_model_to_model(demo_simple_housing: MLPipelineStateModel):
    """

    :return:
    """

    ridge_model = demo_simple_housing.model.to_model()

    assert isinstance(ridge_model, Ridge)


def test_from_model_to_model_params(demo_simple_housing: MLPipelineStateModel):
    """

    :return:
    """

    assert demo_simple_housing.model.params[0].distribution == "log-uniform"

    ridge_model_params = demo_simple_housing.model.get_parameter_space()

    assert isinstance(ridge_model_params, dict)
    assert len(ridge_model_params["alpha"]) == 3
    assert ridge_model_params['alpha'][0] == 1e-16
    assert ridge_model_params['alpha'][1] == 1e16
    assert ridge_model_params['alpha'][2] == "log-uniform"
