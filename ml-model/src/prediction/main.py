import joblib
import os

import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from google.cloud import storage

app = FastAPI()
client = storage.Client()

if "AIP_STORAGE_URI" in os.environ:
    with open("model.joblib", "wb") as f:
        client.download_blob_to_file(f"{os.environ['AIP_STORAGE_URI']}/model.joblib", f)
    _model = joblib.load("model.joblib")
else:
    artifact_path = "./training/model_artifacts"
    _model = joblib.load(f"{artifact_path}/model.joblib")


@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/healthz"))
def health():
    return {}


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
async def predict(request: Request):
    body = await request.json()

    instances = body["instances"]
    inputs_df = pd.DataFrame(instances)
    inputs_df.replace({None: np.nan}, inplace=True)
    outputs = _model.predict(inputs_df).tolist()

    return {"predictions": outputs}