from abc import ABCMeta, abstractmethod
from typing import Dict


class HasParameterSpace(metaclass=ABCMeta):
    """
    This is the trait for creating the parameter space
    """

    @abstractmethod
    def get_parameter_space(self, name: str) -> Dict:
        """
        The abstract method to create the parameter search
        :param name:
        :return:
        """
