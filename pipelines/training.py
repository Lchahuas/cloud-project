import pathlib
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output
from config import TrainingConfig
from vertex_components import upload_model

config = TrainingConfig()

@dsl.container_component
def train(
    train_data: Input[Dataset],
    valid_data: Input[Dataset],
    test_data: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    hparams: dict,
    label: str
):
    return dsl.ContainerSpec(
        image=config.train_container_uri,
        command=["python", "-m"],
        args=[
            "src.main",
            "--train-data",
            train_data.path,
            "--valid-data",
            valid_data.path,
            "--test-data",
            test_data.path,
            "--label",
            label,
            "--model",
            model.path,
            "--metrics",
            metrics.path,
            "--hparams",
            hparams,
        ],
    )

@dsl.pipeline(name=config.pipeline_name)
def pipeline(
    project_id: str = config.project_id,
    project_location: str = config.project_location,
    model_name: str = "lightgbm",
    gcs_train_path: str = config.train_path,
    gcs_valid_path: str = config.valid_path,
    gcs_test_path: str = config.test_path,
):
    
    train_dataset = dsl.importer(
        artifact_uri=gcs_train_path,
        artifact_class=Dataset,
        reimport=True,
    ).set_display_name("Load train data from GCS")
    valid_dataset = dsl.importer(
        artifact_uri=gcs_valid_path,
        artifact_class=Dataset,
        reimport=True,
    ).set_display_name("Load validation data from GCS")

    test_dataset = dsl.importer(
        artifact_uri=gcs_test_path,
        artifact_class=Dataset,
        reimport=True,
    ).set_display_name("Load test data from GCS")

    train_model = train(
        train_data=train_dataset.output,
        valid_data=valid_dataset.output,
        test_data=test_dataset.output,
        hparams=config.hparams,
        label=config.target_column,
    ).set_display_name("Train model")