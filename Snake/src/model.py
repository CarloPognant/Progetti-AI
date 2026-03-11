import torch
import torch.nn as nn
import os


class SnakeNet(nn.Module):
    """
    Dueling DQN:
    - Separa V(s) dal vantaggio A(s,a)
    - Converge più velocemente e in modo più stabile
    """

    def __init__(self, input_size=22, hidden_size=512, output_size=3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_size, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        features  = self.shared(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def save(self, filepath):
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath, device="cpu"):
        self.load_state_dict(torch.load(filepath, map_location=device))