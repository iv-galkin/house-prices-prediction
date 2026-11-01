import pandas as pd

OUTLIER_MIN_AREA = 4500
OUTLIER_MAX_PRICE = 300_000


def load_data(
    train_data_path: str, 
    test_data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает тренировочную и тестовую выборки"""
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


def drop_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дома с огромной площадью и низкой ценой. Только для train."""
    mask = (data["GrLivArea"] > OUTLIER_MIN_AREA) & (data["SalePrice"] < OUTLIER_MAX_PRICE)
    return data[~mask]


def set_categorical_features(
    data: pd.DataFrame,
    categorical_features: list[str],
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Приводит признаки к типу category.

    Если задан reference (обычно train), список категорий берётся из него.
    Значения, которых там нет, становятся NaN.
    """
    source = data if reference is None else reference
    data = data.copy()

    for feature in categorical_features:
        categories = source[feature].astype("category").cat.categories
        data[feature] = pd.Categorical(data[feature], categories=categories)

    return data