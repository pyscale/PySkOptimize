from fastapi import FastAPI
from prometheusrock import PrometheusMiddleware, metrics_route

from api.routes.tasks import router as tasks_router
from api.routes.demo import router as demo_router

api = FastAPI()

api.include_router(tasks_router)
api.include_router(demo_router)

api.add_middleware(
    PrometheusMiddleware,
    skip_paths=['/_healthcheck']
)

api.add_route("/metrics", metrics_route)