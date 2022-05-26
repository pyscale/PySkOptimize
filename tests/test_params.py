from typing import Tuple

import numpy as np

from pyskoptimize.params import DefaultFloatParamModel, DefaultCollectionParamModel, DefaultBooleanParamModel, \
    DefaultStringParamModel, DefaultIntegerParamModel, UniformlyDistributedIntegerParamModel, \
    UniformlyDistributedParamModel, CategoricalParamModel, NormallyDistributedParamModel


def test_default_float_param():
    """

    :return:
    """

    param = DefaultFloatParamModel(name="name", valueFloat=0.0)

    param_static = param.to_param()

    assert param_static == 0.0


def test_default_integer_param():
    """

    :return:
    """

    param = DefaultIntegerParamModel(name="name", valueInt=0)

    param_static = param.to_param()

    assert param_static == 0


def test_default_boolean_param():
    """

    :return:
    """

    param = DefaultBooleanParamModel(name="name", valueBool=True)

    param_static = param.to_param()

    assert param_static


def test_default_str_param():
    """

    :return:
    """

    param = DefaultStringParamModel(name="name", valueStr="dummy")

    param_static = param.to_param()

    assert param_static == "dummy"


def test_default_iter_param():
    """

    :return:
    """

    param = DefaultCollectionParamModel(name="name", valueCollection=[1, 2])

    param_static = param.to_param()

    assert isinstance(param_static, Tuple)
    assert param_static[0] == 1
    assert param_static[1] == 2


def test_uniform_float_param():
    """

    :return:
    """

    param = UniformlyDistributedParamModel(name="name", low=0, high=1)

    param_dynamic = param.to_param()

    values = param_dynamic.rvs(100)

    values = np.array(values)

    assert np.all(values <= 1.) and np.all(0. <= values)


def test_uniform_integer_param():
    """

    :return:
    """

    param = UniformlyDistributedIntegerParamModel(name="name", lowInt=0, highInt=10)

    param_dynamic = param.to_param()

    values = param_dynamic.rvs(100)

    values = np.array(values)

    assert np.all(values <= 10) and np.all(0 <= values)


def test_uniform_categorical_param():
    """

    :return:
    """

    param = CategoricalParamModel(name="name", categories=[1, 2, 3])

    param_dynamic = param.to_param()

    values = param_dynamic.rvs(100)

    values = np.array(values)

    assert np.all(np.isin(values, [1, 2, 3]))


def test_normal_param():
    """

    :return:
    """

    param = NormallyDistributedParamModel(name="name", mu=0., sigma=1.)

    param_dynamic = param.to_param()

    assert param_dynamic[0] == 0.
    assert param_dynamic[1] == 1.
    assert param_dynamic[2] == "normal"


def test_log_normal_param():
    """

    :return:
    """

    param = NormallyDistributedParamModel(name="name", mu=0., sigma=1., log_scale=True)

    param_dynamic = param.to_param()

    assert param_dynamic[0] == 0.
    assert param_dynamic[1] == 1.
    assert param_dynamic[2] == "log-normal"
