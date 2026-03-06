import torch
import torch.nn as nn

class SnakeNet(nn.Module):
    """
    Rete neurale per il gioco Snake.
    Input: 11 valori (stato del gioco)
    Output: 3 azioni (sinistra, dritto, destra)
    """
    def __init__(self, input_size=11, hidden_size=128, output_size=3):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def save(self, filepath):
        """Salva il modello"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """Carica il modello"""
        self.load_state_dict(torch.load(filepath))
