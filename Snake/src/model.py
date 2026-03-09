import torch
import torch.nn as nn
import os


class SnakeNet(nn.Module):
    """
    Rete neurale con architettura Dueling DQN.
    - Separa la stima del valore dello stato (V) dal vantaggio per azione (A)
    - Converge più velocemente e in modo più stabile rispetto al DQN standard
    """

    def __init__(self, input_size=20, hidden_size=512, output_size=3):
        super().__init__()

        # Strati condivisi (feature extractor)
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

        # Stream del valore V(s) — quanto è buono lo stato
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Stream del vantaggio A(s,a) — quanto è buona ogni azione rispetto alla media
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        features  = self.shared(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values  = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def save(self, filepath):
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath, device="cpu"):
        self.load_state_dict(torch.load(filepath, map_location=device))