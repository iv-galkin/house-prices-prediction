import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import build_preprocessor


class MLP(nn.Module):
    """Полносвязная нейросеть для регрессии"""

    def __init__(self, input_dim, hidden_dims=(16,), activation=nn.ReLU, dropout=0.0, batchnorm=False):
        super().__init__()

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(1) # последний слой выдаёт размерность (batch_size, 1), squeeze убирает лишнее измерение, чтобы размерность совпала с y: (batch_size,)


def make_loader(X, y, batch_size, shuffle):
    """Оборачивает признаки и таргет в DataLoader"""
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(np.asarray(y), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, loss_fn):
    """Одна тренировочная эпоха с возвратом среднего rmse по образцам"""
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    return np.sqrt(total_loss / len(loader.dataset))


def evaluate(model, loader, loss_fn):
    """Возвращает средний RMSE модели на валидационной выборке."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            total_loss += loss_fn(preds, yb).item() * len(xb)

    n_samples = len(loader.dataset)
    return np.sqrt(total_loss / n_samples)


def fit_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_dims=(16,),
    activation=nn.ReLU,
    batchnorm=False,
    dropout=0.0,
    optimizer_cls=torch.optim.Adam,
    cosine=False,
    lr=1e-3,
    batch_size=32,
    n_epochs=60,
    random_state=42,
):
    """Обучает нейросеть и возвращает модель и метрики по эпохам"""
    torch.manual_seed(random_state)

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    model = MLP(X_train.shape[1], hidden_dims, activation, dropout, batchnorm)
    nn.init.constant_(model.model[-1].bias, y_train.mean()) # инициализируем bias средним значением y_train, чтобы ускорить сходимость
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs) if cosine else None
    loss_fn = nn.MSELoss()

    history = {"train_rmse": [], "val_rmse": []}
    for _ in range(n_epochs):
        history["train_rmse"].append(train_epoch(model, train_loader, optimizer, loss_fn))
        val_rmse = evaluate(model, val_loader, loss_fn)
        scheduler.step() if scheduler else None
        history["val_rmse"].append(val_rmse)

    return model, history


def cross_validate_mlp(X, y, num_features, cat_features, cv, **fit_params):
    """Возвращает rmse модели на каждом фолде"""
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        preprocessor = build_preprocessor(num_features, cat_features)
        X_train = preprocessor.fit_transform(X.iloc[train_idx])
        X_val = preprocessor.transform(X.iloc[val_idx])

        _, history = fit_mlp(
            X_train, y.iloc[train_idx],
            X_val, y.iloc[val_idx],
            **fit_params,
        )
        scores.append(history["val_rmse"][-1])

    return np.array(scores)
