from enum import Enum

from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field


class ParamType(str, Enum):
    """

    """
    tuple: str = "tuple"
    integer: str = "int"
    double: str = "float"
    categorical: str = "categorical"


class NodeType(str, Enum):
    """

    """
    circle: str = "circle"
    rect: str = "rect"


class DistributionEnum(str, Enum):
    """

    """
    normal: str = "normal"
    log_normal: str = "log-normal"
    uniform: str = "uniform"
    log_uniform: str = "log-uniform"


class SklearnTransformerParamModel(BaseModel):
    """

    """
    type: ParamType
    minValue: Optional[Union[float, int]] = Field(None)
    maxValue: Optional[Union[float, int]] = Field(None)
    distribution: Optional[DistributionEnum] = Field(None)
    categories: Optional[List[str]] = Field(None)


class SklearnTransformerModel(BaseModel):
    """

    """

    name: str
    params: Dict[str, SklearnTransformerParamModel]


class FeaturePeaModel(BaseModel):
    """

    """
    name: str
    type: NodeType


class FeaturePodModel(BaseModel):
    """

    """
    name: str
    features: List[str]
    pipeline: List[SklearnTransformerModel]
    type: NodeType


class MLPipelineStateModel(BaseModel):
    """

    """

    features: Dict[str, FeaturePeaModel]

    preprocess: Dict[str, FeaturePodModel]

    postprocess: Dict[str, FeaturePodModel]

    model: SklearnTransformerModel

    scoring: str
