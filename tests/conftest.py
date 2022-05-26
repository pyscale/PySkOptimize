import pytest

from pyskoptimize.base import MLOptimizer


@pytest.fixture
def demo_simple_housing():
    """

    :return:
    """
    # the JSON file is the needed configuration
    return MLOptimizer.parse_file("tests/data.json")
