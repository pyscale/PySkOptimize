from abc import ABCMeta, abstractmethod
from typing import Union, List, Tuple, Iterable

from pydantic import BaseModel, Field
from skopt.space import Categorical, Integer, Real

Numeric = Union[float, int]


class BaseParamModel(BaseModel, metaclass=ABCMeta):
    """
    The base abstract class for all scikit-learn compatible parameters

    :var name: The name of the parameter
    """
    name: str

    @abstractmethod
    def to_param(self):
        """
        The abstract class to get the parameter space with the current distribution
        :return: The skopt parameter
        """


class DefaultIntegerParamModel(BaseParamModel):
    """
    The class for the default parameter that is an integer
    """

    valueInt: int

    def to_param(self):
        """
        This will create the default parameter for the integer value

        :return:
        """
        return {self.name: self.valueInt}


class DefaultStringParamModel(BaseParamModel):
    """
    The class for the default parameter that is an string
    """

    valueStr: str

    def to_param(self):
        """
        This will create the default parameter for the string value

        :return:
        """
        return {self.name: self.valueStr}


class DefaultBooleanParamModel(BaseParamModel):
    """
    The class for the default parameter that is a boolean
    """

    valueBool: bool

    def to_param(self):
        """
        This will create the default parameter for the boolean value

        :return:
        """
        return {self.name: self.valueBool}


class DefaultFloatParamModel(BaseParamModel):
    """
    The class for the default parameter that is a float
    """

    valueFloat: float

    def to_param(self):
        """
        This will create the default parameter for the float value

        :return:
        """
        return {self.name: self.valueFloat}


class DefaultIterableParamModel(BaseParamModel):
    """
    The class for the default parameter that is an iterable
    """

    valueIterable: Iterable

    def to_param(self):
        """
        This will create the default parameter for the iterable value

        :return:
        """
        return {self.name: self.valueIterable}


class CategoricalParamModel(BaseParamModel):
    """
    The class to handle categorical parameter values

    :var name: The name of the parameter
    :var categories: The list of categories we are going to use
    """
    categories: List

    def to_param(self):
        """
        This converts the param to what skopt needs

        :return: The categorical
        """
        return Categorical(self.categories)


class UniformlyDistributedIntegerParamModel(BaseParamModel):
    """
    This is for the uniform integer distribution
    """
    lowInt: int
    highInt: int

    def to_param(self):
        """
        This converts the param to what skopt needs

        :return: The integer parameter
        """
        return Integer(self.lowInt, self.highInt)


class NumericParamModel(BaseParamModel, metaclass=ABCMeta):
    """
    An abstract base class for all purely numeric parameters

    :var name: The name of the parameter
    :var log_scale: A boolean if we are using the log scale
    """
    log_scale: bool = Field(False)


class UniformlyDistributedParamModel(NumericParamModel):
    """
    The class for uniformly (or log-uniformly) distributed parameters

    :var name: The name of the parameter
    :var low: The lowest value
    :var high: The highest value
    :var log_scale: A boolean if we are using the log scale
    """
    low: Numeric
    high: Numeric

    def to_param(self):
        """
        This converts the param to what skopt needs

        :return: The skopt parameter
        """

        if self.log_scale:
            d = "log-uniform"
        else:
            d = "uniform"

        return Real(
            self.low,
            self.high,
            prior=d
        )


class NormallyDistributedParamModel(NumericParamModel):
    """
    The class for normally (or log-normally) distributed parameters

    :var name: The name of the parameter
    :var mu: The mean
    :var sigma: The variance
    :var log_scale: A boolean if we are using the log scale
    """
    mu: Numeric
    sigma: Numeric

    def to_param(self) -> Tuple:
        """
        This converts the param to what skopt needs

        :return: The skopt parameter
        """

        if self.log_scale:
            d = "log-normal"
        else:
            d = "normal"

        return (
            self.mu,
            self.sigma,
            d
        )
