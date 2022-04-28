from enum import Enum

from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field

Numeric = Union[float, int]


class ParamType(str, Enum):
    """

    """
    numeric: str = "numeric"
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
    boundValues: Union[List[Numeric], List[str], List[bool]]
    distribution: DistributionEnum = Field(DistributionEnum.uniform)


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
