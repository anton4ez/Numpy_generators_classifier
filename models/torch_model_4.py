import torch
import torch.nn as nn


class LSTM_Net(nn.Module):
    def __init__(
            self, 
            lstm_layers = 2, 
            hidden_size = 128, 
            num_classes = 5
        ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=16, padding=8),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:         #(B, 1, 512*512)
            x = x.unsqueeze(1)

        features = self.conv(x)

        features = features.transpose(1, 2)

        lstm_out, _ = self.lstm(features)

        max_hiddens = torch.max(lstm_out, dim=1).values
        mean_hiddens = torch.mean(lstm_out, dim=1)
        std_hiddens = torch.std(lstm_out, dim=1, unbiased=False)

        pooled = torch.cat([mean_hiddens, max_hiddens, std_hiddens], dim=1)

        return self.linear(pooled)
