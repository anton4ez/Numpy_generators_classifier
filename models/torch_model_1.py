import torch.nn as nn

class LSTM_Net(nn.Module):
    def __init__(self, hidden_size=64, num_classes=5, window=2048, stride=2048, input_size=64):
        super().__init__()
        self.window = window
        self.stride = stride
        self.input_size = input_size
        self.seq_len = window // input_size  # 2048 / 64 = 32 шага
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #принимает [B, 128, 2048]
        x = x.unfold(dimension=1, size=self.window, step=self.stride)
        B, N, W = x.shape
        
        #переводим каждое окно размера 2048 в последовательность (32, 64): [B * 128, 32, 64]
        x = x.reshape(B * N, self.seq_len, self.input_size)

        _, (h_n, _) = self.lstm(x)                       # h_n: [1, B * 128, 64]
        h = h_n[-1].reshape(B, N, -1).max(dim=1).values  #  [B, 64]
        return self.fc(h)