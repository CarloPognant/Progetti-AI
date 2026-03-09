"""
Configurazione centralizzata del progetto Snake AI
Modifica i parametri da qui!
"""

import os

# 📁 Percorsi - Calcolati automaticamente
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# Crea le cartelle se non esistono
for directory in [MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

MODEL_BEST_PATH = os.path.join(MODELS_DIR, "snake_model_best.pt")
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, "snake_model_final.pt")
BEST_SCORE_PATH = os.path.join(LOGS_DIR, "best_score.txt")

# 🎮 Impostazioni del Gioco
ROWS = 15
COLS = 17
CELL_SIZE = 40

# 🧠 Iperparametri di Training
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05
BATCH_SIZE = 128
MEMORY_SIZE = 100000
EPISODES = 25000
TARGET_UPDATE = 10

# 🎯 Configurazione Fine-tuning
EPSILON_FINETUNING = 0.2
EPISODES_FINETUNING = 15000

# 🖥️ Configurazione Rendering
FPS = 5

# 🎯 Iperparametri Rete Neurale
INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

# 🔧 Altre configurazioni
USE_GPU = True
VERBOSE = True
LOAD_PREVIOUS_MODEL = True