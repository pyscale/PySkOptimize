from fastapi import APIRouter
from fastapi.responses import JSONResponse
from celery.result import AsyncResult


router = APIRouter(prefix="/tasks")


@router.get("/{task_id}")
def get_task_status(task_id):
    """

    :return:
    """
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }

    return JSONResponse(result)
