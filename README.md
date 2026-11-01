# house-prices-prediction

Предсказание цен домов (Kaggle House Prices). Метрика — RMSE по `log1p(SalePrice)`, 5-кратная кросс-валидация (`KFold`, `random_state: 42`).

Лучшая модель — ансамбль Ridge + XGBoost, CV RMSE 0.1108.

| Модель                               | CV RMSE |
|--------------------------------------|--------:|
| Ансамбль: Ridge + XGBoost            |  0.1108 |
| Ансамбль: Ridge + LightGBM + XGBoost |  0.1115 |
| Ансамбль: Ridge + LightGBM           |  0.1117 |
| MLP                                  |  0.1127 |
| Ridge                                |  0.1138 |
| Ridge с кодированием качества        |  0.1142 |
| Ridge без составных площадей         |  0.1145 |
| XGBoost                              |  0.1174 |
| LightGBM                             |  0.1184 |
| Random Forest, one-hot               |  0.1290 |
| Random Forest, ordinal               |  0.1292 |
| KNN                                  |  0.1694 |

На public leaderboard у ансамбля Ridge + XGBoost скор 0.12620.

`main.py` читает `config.yaml`, обучает Ridge и XGBoost и пишет `output/submission.csv`. Параметры моделей в конфиге это лучшие значения из `02_classical_ml.ipynb`.

- `data` — пути к `input_files/train.csv`, `input_files/test.csv` и `output/submission.csv`
- `features.drop` — признаки, которые не попадают в модель, остальные делятся на числовые и категориальные по типу
- `model` — параметры Ridge и XGBoost

## Запуск

```bash
uv sync
uv run python main.py
```

Ноутбуки:

- `notebooks/01_eda.ipynb` — разведочный анализ
- `notebooks/02_classical_ml.ipynb` — выбор признаков и сравнение Ridge, KNN, Random Forest, LightGBM, XGBoost и ансамбля
- `notebooks/03_nn.ipynb` — эксперименты с полносвязной нейросетью
