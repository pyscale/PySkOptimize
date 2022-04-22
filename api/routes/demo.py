
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..models import MLPipelineStateModel
from api.utils.tasks import task_housing_demo

router = APIRouter(prefix="/demos")


@router.post("/housing")
def california_housing(request_model: MLPipelineStateModel):
    """

    :return:
    """
    task = task_housing_demo.delay(request_model.dict())

    return JSONResponse({'task_id': task.id})
    