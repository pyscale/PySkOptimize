from api.utils.ml.utils import from_request_to_model
from api.utils.ml.train import training_housing_model


def test_training_housing_model(demo_simple_housing):
    """

    :param demo_simple_housing:
    :return:
    """
    model = from_request_to_model(demo_simple_housing)

    res = training_housing_model(
        model, demo_simple_housing.scoring
    )

    assert 0.5 < abs(res.testing_score)
