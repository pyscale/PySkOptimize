
from fastapi import APIRouter
from ..models import MLPipelineStateModel
from ..utils.ml.utils import from_request_to_model

router = APIRouter(prefix="/demos")


@router.post("/housing")
def california_housing(request_model: MLPipelineStateModel):
    """

    :return:
    """

    model = from_request_to_model(request_model)

    