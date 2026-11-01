from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data import drop_outliers, load_data, set_categorical_features
from src.features import add_features, clean_data
from src.preprocessing import build_preprocessor, split_features


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Cleaning train and test datasets")
    train = clean_data(train)
    test = clean_data(test)

    logger.info("Adding features to train and test datasets")
    train = add_features(train)
    test = add_features(test)

    logger.info("Dropping outliers from train dataset")
    train = drop_outliers(train)
    return train, test


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> np.ndarray:
    logger.info("Loading features")
    num_features, cat_features = split_features(
        train,
        config["features"]["drop"],
        config["target"],
    )
    features = num_features + cat_features
    X_train = train[features]
    y_train = np.log1p(train[config["target"]])
    X_test = test[features]

    ridge = Pipeline([
        ("preprocessing", build_preprocessor(num_features, cat_features)),
        ("ridge", Ridge(alpha=config["model"]["ridge"]["alpha"])),
    ])
    xgboost = XGBRegressor(**config["model"]["xgboost"])

    X_train_boost = set_categorical_features(X_train, cat_features)
    X_test_boost = set_categorical_features(X_test, cat_features, reference=X_train_boost)

    logger.info("Fitting ridge model")
    ridge.fit(X_train, y_train)
    logger.info("Fitting xgboost model")
    xgboost.fit(X_train_boost, y_train)

    logger.info("Predicting on test dataset")
    ridge_pred = ridge.predict(X_test)
    xgboost_pred = xgboost.predict(X_test_boost)
    return np.mean([ridge_pred, xgboost_pred], axis=0)


def save_submission(
    test: pd.DataFrame,
    log_preds: np.ndarray,
    config: dict,
    path: Path,
) -> None:
    submission = pd.DataFrame({
        config["id_column"]: test[config["id_column"]],
        config["target"]: np.expm1(log_preds),
    })
    submission.to_csv(path, index=False)
    logger.success("Submission saved to {}", config["data"]["submission_path"])


def main() -> None:
    root = Path(__file__).parent
    config = load_config(root / "config.yaml")

    logger.add(
        root / "output" / "pipeline.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    logger.info("Loading train and test datasets")
    train, test = load_data(
        root / config["data"]["train_path"],
        root / config["data"]["test_path"],
    )
    logger.info("Loaded {} train rows and {} test rows", len(train), len(test))

    train, test = prepare(train, test)
    log_preds = fit_predict(train, test, config)
    save_submission(test, log_preds, config, root / config["data"]["submission_path"])


if __name__ == "__main__":
    main()
