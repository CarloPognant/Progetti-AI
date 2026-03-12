import torch
import torch.nn as nn
import os


class SnakeNet(nn.Module):
    """
    Dueling DQN con backbone CNN.

    Input: tensore (batch, 3, ROWS, COLS)
      - Canale 0: corpo del serpente con gradiente (testa=1.0, coda→0.0)
                  → la rete "vede" la forma e la direzione del corpo
      - Canale 1: testa del serpente (1.0 solo sulla cella testa)
      - Canale 2: mela (1.0 solo sulla cella mela)

    Perché CNN invece dei 22 feature?
      Con 22 feature il serpente riceveva solo informazioni locali (celle adiacenti,
      distanze scalari). Non poteva "vedere" il proprio corpo a distanza, quindi
      con 40+ segmenti non riusciva a pianificare percorsi sicuri.
      La CNN processa l'intera griglia → il serpente ha visione globale.
    """

    def __init__(self, rows=15, cols=17, hidden_size=512, output_size=3):
        super().__init__()

        self.rows = rows
        self.cols = cols

        # ── Backbone CNN ──────────────────────────────────────────
        # 3 conv layer con padding=1 → mantengono la dimensione spaziale
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        cnn_flat = 64 * rows * cols   # es. 64 * 15 * 17 = 16320

        # ── Fully connected condiviso ─────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(cnn_flat, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
        )

        # ── Dueling heads ─────────────────────────────────────────
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
        # x: (batch, 3, rows, cols)
        features  = self.cnn(x)
        features  = features.view(features.size(0), -1)   # flatten
        features  = self.shared(features)
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