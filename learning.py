from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score
from tqdm import tqdm

from H5PY.reader import HDF5Reader, InMemoryHDF5Reader
from models.torch_model_4 import LSTM_Net


MODEL_NUM = 4


def ensure_directories(base_dir: Path) -> Tuple[Path, Path]:
    """
    Создает необходимые директории для чекпоинтов и предсказаний.

    :param base_dir: Корневой путь для сохранения артефактов конкретной модели.
    :return: Кортеж из путей (директория моделей, директория предсказаний).
    """
    models_dir = base_dir / "models_for_epoch"
    predicts_dir = base_dir / "predicts_for_epoch"
    models_dir.mkdir(parents=True, exist_ok=True)
    predicts_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, predicts_dir


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_norm: float = 1.0,
    epoch_idx: int = 1,
    total_epochs: int = 100,
) -> float:
    """
    Выполняет одну эпоху обучения модели.

    :param model: Обучаемая нейросеть.
    :param dataloader: Загрузчик обучающих данных.
    :param criterion: Функция потерь.
    :param optimizer: Оптимизатор весов.
    :param device: Устройство вычислений (CPU или CUDA).
    :param clip_norm: Максимальная норма градиентов для clipping.
    :param epoch_idx: Текущий номер эпохи (для отображения).
    :param total_epochs: Общее число эпох.
    :return: Среднее значение функции потерь на обучении.
    """
    model.train()
    total_loss = 0.0

    desc = f"Train Epoch {epoch_idx}/{total_epochs}"
    for x_batch, y_batch in tqdm(dataloader, desc=desc, leave=False):
        x_batch = x_batch.to(device, non_blocking=True).float()
        y_batch = y_batch.to(device, non_blocking=True)

        x_batch = (x_batch - 127.5) / 127.5

        optimizer.zero_grad()
        predict = model(x_batch)
        loss = criterion(predict, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    f1_metric: MulticlassF1Score,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Выполняет валидацию модели на тестовом наборе данных.

    :param model: Оцениваемая нейросеть.
    :param dataloader: Загрузчик тестовых данных.
    :param criterion: Функция потерь.
    :param f1_metric: Метрика Macro F1-score.
    :param device: Устройство вычислений.
    :return: Кортеж (средний test_loss, test_f1, массив всех логитов/вероятностей, массив меток).
    """
    model.eval()
    f1_metric.reset()
    total_loss = 0.0

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for x_test, y_test in dataloader:
            x_test = x_test.to(device, non_blocking=True).float()
            y_test = y_test.to(device, non_blocking=True)

            predict = model(x_test)
            total_loss += criterion(predict, y_test).item()
            f1_metric.update(predict, y_test)

            all_preds.append(predict.detach().cpu().numpy())
            all_targets.append(y_test.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1_score = float(f1_metric.compute().item())
    
    # Объединяем батчи в единые массивы за один проход
    stacked_preds = np.concatenate(all_preds, axis=0)
    stacked_targets = np.concatenate(all_targets, axis=0)

    return avg_loss, f1_score, stacked_preds, stacked_targets


def save_epoch_predictions(file_path: Path, preds: np.ndarray, targets: np.ndarray) -> None:
    """
    Сохраняет предсказания модели и истинные метки за эпоху в HDF5.

    :param file_path: Полный путь к создаваемому файлу HDF5.
    :param preds: Массив предсказаний размерности (N, num_classes).
    :param targets: Вектор меток размерности (N,).
    """
    with h5py.File(file_path, "w") as f:
        f.create_dataset("X_predicted", data=preds, dtype=np.float64, chunks=True)
        f.create_dataset("Y", data=targets, dtype=np.float64, chunks=True)


def save_checkpoints(
    models_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    is_best: bool,
) -> None:
    """
    Сохраняет контрольную точку эпохи и обновляет best_model.pth при необходимости.

    :param models_dir: Папка для сохранения моделей.
    :param epoch: Номер текущей эпохи.
    :param model: Экземпляр нейросети.
    :param optimizer: Оптимизатор.
    :param is_best: Флаг лучшего результата по валидационной метрике.
    """
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
    }
    torch.save(checkpoint, models_dir / f"model_epoch_{epoch}.tar")

    if is_best:
        torch.save(model.state_dict(), models_dir / "best_model.pth")


def save_run_summary(
    file_path: Path,
    loss_train: List[float],
    loss_test: List[float],
    f1_test: List[float],
) -> None:
    """
    Сохраняет итоговую статистику обучения (loss, f1) за все эпохи в HDF5.

    :param file_path: Путь к файлу info.hdf5.
    :param loss_train: Список значений потерь на обучении.
    :param loss_test: Список значений потерь на валидации.
    :param f1_test: Список значений F1-score на валидации.
    """
    epochs_count = len(loss_train)
    data = np.column_stack([loss_train, loss_test, f1_test]).astype(np.float64)

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(file_path, "w") as f:
        dset = f.create_dataset("info", data=data)
        dset.attrs.create(
            "columns", ["loss_train", "loss_test", "f1_test"], dtype=str_dt
        )
        dset.attrs.create(
            "rows", [f"epoch_{i}" for i in range(1, epochs_count + 1)], dtype=str_dt
        )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_num = MODEL_NUM
    base_checkpoint_dir = Path(f"checkpoints/model_{model_num}")
    models_dir, predicts_dir = ensure_directories(base_checkpoint_dir)

    model = LSTM_Net(lstm_layers = 2, 
                     hidden_size = 128, 
                     num_classes = 5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=0.0005,
                                    alpha=0.99,
                                    momentum=0.9,
                                    weight_decay=1e-4
    )
    f1_metric = MulticlassF1Score(num_classes=5, average="macro").to(device)

    #/data/anton4ez/ для кластера
    train_dataset = InMemoryHDF5Reader("/data/anton4ez/train_dataset.hdf5")
    test_dataset  = InMemoryHDF5Reader("/data/anton4ez/test_dataset.hdf5")

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset, 
        batch_size=256, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    total_epochs = 100
    best_f1      = -float("inf")
    best_epoch   = 0

    history = {
        "loss_train": [],
        "loss_test": [],
        "f1_test": [],
    }

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            clip_norm=1.0,
            epoch_idx=epoch,
            total_epochs=total_epochs,
        )

        test_loss, test_f1, preds, targets = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            f1_metric=f1_metric,
            device=device,
        )

        pred_file = predicts_dir / f"predicts_epoch_{epoch}.hdf5"
        save_epoch_predictions(pred_file, preds, targets)

        is_best = test_f1 > best_f1
        if is_best:
            best_f1 = test_f1
            best_epoch = epoch

        save_checkpoints(models_dir, epoch, model, optimizer, is_best)

        history["loss_train"].append(train_loss)
        history["loss_test"].append(test_loss)
        history["f1_test"].append(test_f1)

        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test F1: {test_f1:.4f}"
            + (" [BEST]" if is_best else "")
        )

    print(f"\nTraining completed. Best F1: {best_f1:.4f} at epoch {best_epoch}")

    save_run_summary(
        file_path=base_checkpoint_dir / "info.hdf5",
        loss_train=history["loss_train"],
        loss_test=history["loss_test"],
        f1_test=history["f1_test"],
    )


if __name__ == "__main__":
    main()
