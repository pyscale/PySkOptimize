import pytest

from pyskoptimize import MLPipelineStateModel


@pytest.fixture
def demo_simple_housing():
    """

    :return:
    """
    # the JSON file is the needed configuration
    return MLPipelineStateModel.parse_file("tests/data.json")
