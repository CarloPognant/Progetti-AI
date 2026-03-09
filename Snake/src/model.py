import torch
import torch.nn as nn
import os
import sys

class SnakeNet(nn.Module):
    """
    Rete neurale MIGLIORATA per Snake AI
    - Input: 11 valori (stato)
    - Output: 3 azioni
    """
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def save(self, filepath):
        """Salva il modello"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath, device="cpu"):
        """Carica il modello"""
        self.load_state_dict(torch.load(filepath, map_location=device))