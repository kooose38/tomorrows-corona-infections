import torch.nn as nn 
import torch 
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, hidden_dim, input_size=6, tag_size=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tag_size)
    
    def forward(self, x):
        # x.size (1, 30日分のデータ、各特長量6)
        hs, _ = self.lstm(x, None)
        out = hs[:, -1, :] # 時系列に従い最後の隠れ層を使う
        out = F.hardtanh(self.fc(out), 0, 80000)
        return out 

