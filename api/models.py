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
    name: str
    type: ParamType
    minValue: Optional[Union[float, int]] = Field(None)
    maxValue: Optional[Union[float, int]] = Field(None)
    distribution: Optional[DistributionEnum] = Field(None)
    categories: Optional[List[str]] = Field(None)


class SklearnTransformerModel(BaseModel):
    """

    """

    name: str
    params: Optional[List[SklearnTransformerParamModel]] = Field(None)


class FeaturePodModel(BaseModel):
    """

    """
    name: str
    features: List[str]
    pipeline: List[SklearnTransformerModel]


class MLPipelineStateModel(BaseModel):
    """

    """

    preprocess: List[FeaturePodModel]

    postprocess: Optional[List[FeaturePodModel]]

    model: SklearnTransformerModel

    scoring: str
