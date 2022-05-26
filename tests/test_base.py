from pyskoptimize.steps import SklearnTransformerModel
from sklearn.linear_model import Ridge


def test_transformer_model_parse():
    """

    :return:
    """
    sk_transformer_model = SklearnTransformerModel(name="sklearn.linear_model.Ridge")

    assert isinstance(sk_transformer_model.to_model(), Ridge)


def test_transformer_model_none_params_parse():
    """

    :return:
    """
    sk_transformer_model = SklearnTransformerModel(name="sklearn.linear_model.Ridge")

    assert len(list(sk_transformer_model.get_parameter_space().keys())) == 0


def test_transformer_model_none_default_params_parse():
    """

    :return:
    """
    sk_transformer_model = SklearnTransformerModel(name="sklearn.linear_model.Ridge")

    assert len(list(sk_transformer_model.get_default_parameter_space().keys())) == 0
