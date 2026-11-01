import pandas as pd


# NA в этих признаках означает отсутствие объекта
NONE_CATEGORIES = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

# Площади и количества, которые равны 0, если объекта нет
ZERO_NUMERIC = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]


def fill_none_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Заполняет 'None' категориальные признаки, где NA означает отсутствие объекта."""
    data = data.copy()
    data[NONE_CATEGORIES] = data[NONE_CATEGORIES].fillna("None")
    return data


def fill_zero_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """Заполняет нулём площади и количества у домов без подвала, гаража или облицовки."""
    data = data.copy()
    data[ZERO_NUMERIC] = data[ZERO_NUMERIC].fillna(0)
    return data


def set_mssubclass_category(data: pd.DataFrame) -> pd.DataFrame:
    """Переводит MSSubClass в строку: это код типа дома, а не число."""
    data = data.copy()
    data["MSSubClass"] = data["MSSubClass"].astype(str)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Заполняет смысловые пропуски и исправляет типы признаков."""
    data = fill_none_categories(data)
    data = fill_zero_numeric(data)
    data = set_mssubclass_category(data)
    return data


def add_total_sf(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет TotalSF — общая площадь дома: подвал + первый + второй этаж."""
    data = data.copy()
    data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
    return data


def add_total_bath(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет TotalBath — число ванных, половинные считаются за 0.5."""
    data = data.copy()
    data["TotalBath"] = (
        data["FullBath"] + 0.5 * data["HalfBath"]
        + data["BsmtFullBath"] + 0.5 * data["BsmtHalfBath"]
    )
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Добавляет все новые признаки."""
    data = add_total_sf(data)
    data = add_total_bath(data)
    return data


QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

QUALITY_FEATURES = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


def encode_quality(data: pd.DataFrame) -> pd.DataFrame:
    """Кодирует оценки качества числами по порядку: None=0, Po=1, ..., Ex=5."""
    data = data.copy()
    for feature in QUALITY_FEATURES:
        data[feature] = data[feature].map(QUALITY_MAP)
    return data
