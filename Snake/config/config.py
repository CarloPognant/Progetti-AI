"""
Configurazione centralizzata del progetto Snake AI
"""

import os

# 📁 Percorsi
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")

for directory in [MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

MODEL_BEST_PATH  = os.path.join(MODELS_DIR, "snake_model_best.pt")
MODEL_FINAL_PATH = os.path.join(MODELS_DIR, "snake_model_final.pt")
BEST_SCORE_PATH  = os.path.join(LOGS_DIR,   "best_score.txt")

# 🎮 Gioco
ROWS      = 15
COLS      = 17
CELL_SIZE = 40

# 🧠 Iperparametri Training
LEARNING_RATE  = 0.0001
GAMMA          = 0.99
EPSILON        = 1.0          # riparte da 1.0: nuova architettura = nuovo training
EPSILON_DECAY  = 0.9995
EPSILON_MIN    = 0.01
BATCH_SIZE     = 512
MEMORY_SIZE    = 200000
EPISODES       = 50000
TARGET_UPDATE  = 5

# 🚀 Ambienti paralleli
NUM_ENVS = 64

# 🎯 Fine-tuning
EPSILON_FINETUNING  = 0.1
EPISODES_FINETUNING = 10000

# 🖥️ Rendering
FPS = 10

# 🎯 Rete Neurale CNN
# Lo stato non è più un vettore di 22 feature ma una griglia (3, ROWS, COLS)
CNN_CHANNELS = 3       # corpo-gradiente, testa, mela
HIDDEN_SIZE  = 512
OUTPUT_SIZE  = 3

# 🔧 Misc
USE_GPU             = True
VERBOSE             = True
LOAD_PREVIOUS_MODEL = False   # False: architettura cambiata, modello vecchio incompatibile