
from fastapi.testclient import TestClient

from time import sleep


def test_california_housing(client: TestClient, housing_ml_pipeline_state):

    response = client.post("/demos/housing", json=housing_ml_pipeline_state)

    assert response.status_code == 200, response.json()['detail']

    task_id = response.json()['task_id']

    while True:
        response = client.get(f"/tasks/{task_id}")

        assert response.status_code == 200

        task_status = response.json()['task_status']

        if task_status == "SUCCESS":
            break

        else:
            print("Sleeping")
            sleep(5)
