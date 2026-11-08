import torch
import torch.nn as nn


class LSTM_Net(nn.Module):
    def __init__(self, input_size: int = 512, hidden_size: int = 64, num_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 512, 512)

        _, (h_n, _) = self.lstm(x)

        return self.fc(h_n[-1])