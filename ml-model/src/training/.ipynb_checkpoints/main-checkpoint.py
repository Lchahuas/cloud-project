import argparse
import json
import os
import logging
from sklearn.pipeline import Pipeline
from google.cloud import storage
from feature_constants import NUM_COLS, OHE_COLS
from utils import save_metrics, save_model_artifact, save_training_dataset_metadata
from utils import read_datasets, split_xy, indices_in_list

from ml_pipeline import (
    build_ml_pipeline,
    fit_ml_pipeline,
    evaluate_model
)
from transformers import create_model, build_preprocessing_transformer

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--valid-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--model", default=os.getenv("AIP_MODEL_DIR"), type=str, help="")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--hparams", default={}, type=json.loads)
    args = parser.parse_args()

    logging.info("Read csv files into dataframes")
    df_train, df_valid, df_test = read_datasets(args.train_data, args.valid_data, args.test_data)

    logging.info("Split dataframes")
    label = args.label
    X_train, y_train = split_xy(df_train, label)
    X_valid, y_valid = split_xy(df_valid, label)
    X_test, y_test = split_xy(df_test, label)

    logging.info("Get indices of columns in base data")
    col_list = X_train.columns.tolist()
    columns_indices = {
    "numerical": indices_in_list(NUM_COLS, col_list),
    "categorical": indices_in_list(OHE_COLS, col_list),
    }

    logging.info("Build sklearn preprocessing steps")
    preprocesser = build_preprocessing_transformer(columns_indices)

    logging.info("Build sklearn pipeline with LGBMClassifier model")
    lgb_model = create_model(args.hparams)
    pipeline = build_ml_pipeline(preprocessor=preprocesser, model=lgb_model)

    logging.info("Fit ml pipeline")
    fit_ml_pipeline(pipeline, X_train, y_train, X_valid, y_valid, "auc")

    logging.info("Evaluate model")
    metrics = evaluate_model(pipeline, X_test, y_test)

    logging.info(f"Save model to: {args.model}")
    save_model_artifact(pipeline, args.model)

    logging.info(f"Metrics: {metrics}")
    save_metrics(metrics, args.metrics)

    logging.info(f"Persist URIs of training file(s) for model monitoring in batch predictions")
    save_training_dataset_metadata(args.model, args.train_data, label)


if __name__ == "__main__":
    main()