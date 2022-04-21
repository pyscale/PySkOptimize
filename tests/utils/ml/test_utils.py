from api.utils.ml.utils import from_model_to_model, \
    from_model_to_model_params

from api.models import MLPipelineStateModel

from sklearn.linear_model import Ridge


def test_from_model_to_model(demo_simple_housing: MLPipelineStateModel):
    """

    :return:
    """

    ridge_model = from_model_to_model(demo_simple_housing.model)

    assert isinstance(ridge_model, Ridge)


def test_from_model_to_model_params(demo_simple_housing: MLPipelineStateModel):
    """

    :return:
    """

    ridge_model_params = from_model_to_model_params(demo_simple_housing.model.params)

    assert isinstance(ridge_model_params, dict)
    assert ridge_model_params['alpha'][0] == 1e-16
    assert ridge_model_params['alpha'][1] == 1e16
    assert ridge_model_params['alpha'][2] == "log-uniform"
