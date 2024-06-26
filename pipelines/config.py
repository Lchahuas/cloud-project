from os import environ as env

from dataclasses import dataclass


@dataclass(init=False, repr=True, frozen=True)
class Config:
    """Configuration for all pipelines."""

    project_id = env["VERTEX_PROJECT_ID"]
    project_location = env.get("VERTEX_LOCATION")
    solution_name = env.get("SOLUTION_NAME")
    model_name = "lightgbm"
    solution_model_name = f"{solution_name}-{model_name}"


@dataclass(init=False, repr=True, frozen=True)
class TrainingConfig(Config):
    """Configuration for training pipeline."""

    pipeline_name = Config.solution_model_name + "-train"
    target_column = "HeartDisease"

    primary_metric = "auRoc"
    hparams = dict()
    container_image_registry = env["CONTAINER_IMAGE_REGISTRY"]
    train_container_uri = "us-central1-docker.pkg.dev/durable-ring-405419/vertex-images/heart-prediction-src-training:latest"
    serving_container_uri = "us-central1-docker.pkg.dev/durable-ring-405419/vertex-images/heart-prediction-src-prediction:latest"
    serving_container_predict_route = "/predict"
    serving_container_health_route = "/health"
    train_path = "gs://durable-ring-405419_cloudbuild/train.csv"
    test_path = "gs://durable-ring-405419_cloudbuild/test.csv"
    valid_path = "gs://durable-ring-405419_cloudbuild/valid.csv"


@dataclass(init=False, repr=True, frozen=True)
class PredictionConfig(Config):
    """Configuration for prediction pipeline."""

    pipeline_name = Config.solution_model_name + "-predict"
    min_replicas = 30
    max_replicas = 35
    batch_size = 1000
    monitoring_skew_config = {"defaultSkewThreshold": {"value": 0.001}}
    instance_config = {}