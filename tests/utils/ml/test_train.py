from api.utils.ml.train import training_housing_model


def test_training_housing_model(demo_simple_housing):
    """

    :param demo_simple_housing:
    :return:
    """
    model = demo_simple_housing.to_bayes_opt()

    res = training_housing_model(
        model
    )

    assert 0.5 < abs(res.testing_score)
