import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from typing import List, Tuple, Union, Dict
from utilities.custom_transformers import StringContainsTransformer


def build_numeric_transformers(column_indices: Dict[str, List[int]]) -> List[Tuple[str, Union[Pipeline, TransformerMixin], List[int]]]:
    numeric_transformers = [
        (
            "impute_numerical",
            SimpleImputer(strategy="mean"),
            column_indices["numerical"],
        ),
        (
            "scale_numerical",
            StandardScaler(),
            column_indices["numerical"],
        ),
    ]
    return numeric_transformers

def build_categorical_transformers(column_indices: Dict[str, List[int]]) -> List[Tuple[str, Union[Pipeline, TransformerMixin], List[int]]]:
    categorical_transformers = [
        (
            "one_hot_encoding",
            OneHotEncoder(handle_unknown="ignore"),
            column_indices["categorical"],
        ),
    ]
    return categorical_transformers


def build_preprocessing_transformer(column_indices: Dict[str, List[int]]) -> ColumnTransformer:
    numeric_transformers = build_numeric_transformers(column_indices)
    categorical_transformers = build_categorical_transformers(column_indices)

    all_transformers = numeric_transformers + categorical_transformers
    preprocesser = ColumnTransformer(transformers=all_transformers)

    return preprocesser


def create_model(hyperparameters: dict) -> lgb.LGBMClassifier:
    lgb_model = lgb.LGBMClassifier(**hyperparameters)
    return lgb_model