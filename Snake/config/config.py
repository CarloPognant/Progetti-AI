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
EPSILON        = 0.4
EPSILON_DECAY  = 0.9998
EPSILON_MIN    = 0.05
BATCH_SIZE     = 1024     # ↑ aumentato da 512
MEMORY_SIZE    = 200000
EPISODES       = 30000
TARGET_UPDATE  = 5

# 🚀 Ambienti paralleli
NUM_ENVS = 64             # ↑ aumentato da 16

# 🎯 Fine-tuning
EPSILON_FINETUNING  = 0.1
EPISODES_FINETUNING = 10000

# 🖥️ Rendering
FPS = 10

# 🎯 Rete Neurale
INPUT_SIZE  = 22
HIDDEN_SIZE = 512
OUTPUT_SIZE = 3

# 🔧 Misc
USE_GPU             = True
VERBOSE             = True
LOAD_PREVIOUS_MODEL = True