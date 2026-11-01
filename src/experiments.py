"""Общий код для ноутбуков с экспериментами."""

import json
from pathlib import Path

import numpy as np
import optuna
import yaml
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing import split_features, build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "artifacts/best_params.json"

RANDOM_STATE = 42

cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)


def load_features(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Возвращает числовые и категориальные признаки с учётом списка drop из config.yaml."""
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    return split_features(data, config["features"]["drop"], config["target"])


def cv_rmse(model, X, y, n_jobs=None, **fit_params):
    """Возвращает RMSE по каждому фолду разбиения."""
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        params=fit_params or None,
        n_jobs=n_jobs,
        error_score="raise",
    )
    return -scores


def tune_ridge(X, y, num_features, cat_features, alphas=np.logspace(-2, 3, 21)):
    """Подбирает alpha для Ridge. Возвращает alpha, CV RMSE и его std по фолдам."""
    model = Pipeline([
        ("preprocessing", build_preprocessor(num_features, cat_features)),
        ("ridge", Ridge()),
    ])
    search = GridSearchCV(
        model,
        {"ridge__alpha": alphas},
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X, y)
    best = search.best_index_
    return (
        search.best_params_["ridge__alpha"],
        -search.best_score_,
        search.cv_results_["std_test_score"][best],
    )


def run_study(objective, study_name, n_trials=50, params_key=None):
    """Минимизировать objective с помощью TPE-сэмплера с фиксированным seed и вывести лучший результат.

    Если задан params_key, лучшие параметры сохраняются в PARAMS_PATH под этим ключом.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best CV RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    if params_key is not None:
        save_best_params(params_key, study.best_params)
    return study


def save_best_params(name, params):
    """Дописать лучшие параметры модели в общий JSON."""
    all_params = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    all_params[name] = params
    PARAMS_PATH.parent.mkdir(exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(all_params, indent=2))


def load_best_params(name):
    """Прочитать сохранённые лучшие параметры модели."""
    return json.loads(PARAMS_PATH.read_text())[name]
