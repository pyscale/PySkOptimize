from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd


def test_training_housing_model(demo_simple_housing):
    """
    This tests we can achieve a predictable performance

    :param demo_simple_housing:
    :return:
    """
    cal_housing = fetch_california_housing()
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)

    model = demo_simple_housing.to_bayes_opt(verbose=3)

    model.fit(
        X_train,
        y_train
    )

    testing_score = model.score(X_test, y_test)

    assert 0.5 < abs(testing_score)
