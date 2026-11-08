import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Остаточный свёрточный блок для выявления локальных закономерностей байтового потока."""
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FeatureEvolverLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        num_classes: int = 5,
        lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 1. Модуль выращивания локальных признаков
        # Вход: (B, 1, 262144). Сжимаем длину и растим пространство каналов.
        self.feature_extractor = nn.Sequential(
            # (B, 1, 262144) -> (B, 32, 65536)
            nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            ConvBlock(32),

            # (B, 32, 65536) -> (B, 64, 8192)
            nn.Conv1d(32, 64, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ConvBlock(64),

            # (B, 64, 8192) -> (B, 128, 512)
            # Получаем 512 временных шагов, на каждом шаге 128 сгенерированных фичей
            nn.Conv1d(64, 128, kernel_size=32, stride=16, padding=8),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # 2. Основной рекуррентный блок
        # На вход подается 512 шагов по 128 фичей
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,  # Смотрит контекст в обе стороны
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # 3. Классификатор с агрегацией по времени
        # bidirectional=True удваивает hidden_size: (hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Приводим к виду (B, 1, 262144)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.view(x.size(0), 1, -1)

        # Выращивание признаков: (B, 1, 262144) -> (B, 128, 512)
        feats = self.feature_extractor(x)

        # Подготовка под LSTM: (B, channels, seq_len) -> (B, seq_len, channels)
        # Получаем (B, 512, 128)
        feats = feats.transpose(1, 2)

        # Прогон через LSTM: out shape [B, 512, hidden_size * 2]
        lstm_out, _ = self.lstm(feats)

        # Усредняем скрытые состояния по всем 512 временным шагам (Mean Pooling)
        # Это эффективнее простого взятия h_n[-1], так как учитывает всю траекторию генератора
        pooled = torch.mean(lstm_out, dim=1)

        # Логиты классов
        return self.classifier(pooled)