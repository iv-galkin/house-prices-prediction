import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_preprocessor(
    num_features: list[str],
    cat_features: list[str],
) -> ColumnTransformer:
    """Препроцессинг числовых и категориальных фичей.
    
    Числовые фичи заполняются медианой и стандартизуются.
    Категориальные заполняются модой и кодируются one-hot.
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])


def build_tree_preprocessor(
    num_features: list[str],
    cat_features: list[str],
) -> ColumnTransformer:
    """Препроцессинг для деревьев: без масштабирования, категории кодируются числами."""
    num_pipe = SimpleImputer(strategy="median")

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])


def split_features(
    data: pd.DataFrame,
    drop: list[str],
    target: str,
) -> tuple[list[str], list[str]]:
    data = data.drop(columns=drop + [target], errors="ignore")
    num_features = data.select_dtypes(include="number").columns.tolist()
    cat_features = data.select_dtypes(include="object").columns.tolist()
    return num_features, cat_features
