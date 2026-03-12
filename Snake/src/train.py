import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import sys
import numpy as np
from collections import deque
from datetime import datetime
import time

sys.stdout.flush()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from sumtree import PrioritizedReplayBuffer

from model     import SnakeNet
from snake_env import SnakeAIEnv
from config    import (
    LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN,
    BATCH_SIZE, MEMORY_SIZE, EPISODES, TARGET_UPDATE,
    MODEL_BEST_PATH, MODEL_FINAL_PATH, BEST_SCORE_PATH,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    LOAD_PREVIOUS_MODEL, NUM_ENVS, MODELS_DIR, LOGS_DIR,
)

def print_flush(msg):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════
#  📝 LOG FILE
# ══════════════════════════════════════════════════════════════

LOG_FILE_PATH = os.path.join(LOGS_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(msg):
    """Stampa a terminale E scrive sul file di log."""
    print_flush(msg)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ══════════════════════════════════════════════════════════════
#  🎨 CONFIGURAZIONE LOGGER AVANZATO
# ══════════════════════════════════════════════════════════════

LOG_INTERVAL = 100  # Log ogni 100 episodi reali

class TrainingLogger:
    def __init__(self):
        self.start_time      = time.time()
        self.last_log_time   = time.time()
        self.last_log_episode = 0

    def should_log(self, current_episode):
        return current_episode - self.last_log_episode >= LOG_INTERVAL

    def log_progress(self, episode, best_score, avg_score, epsilon, memory_size):
        current_time       = time.time()
        elapsed_total      = current_time - self.start_time
        elapsed_since_log  = current_time - self.last_log_time
        episodes_since_log = episode - self.last_log_episode

        eps_per_sec = episodes_since_log / elapsed_since_log if elapsed_since_log > 0 else 0

        hours   = int(elapsed_total // 3600)
        minutes = int((elapsed_total % 3600) // 60)
        seconds = int(elapsed_total % 60)

        log(f"\n{'='*70}")
        log(f"📊 EPISODIO {episode:,} | ⏱️  {hours:02d}:{minutes:02d}:{seconds:02d}")
        log(f"{'='*70}")
        log(f"  🏆 Best Score     : {best_score:3d} mele")
        log(f"  📈 Avg (100 ep)   : {avg_score:5.2f} mele")
        log(f"  🎲 Epsilon        : {epsilon:.4f} ({(1-epsilon)*100:.1f}% sfruttamento)")
        log(f"  💾 Memory size    : {memory_size:,} / {MEMORY_SIZE:,}")
        log(f"  ⚡ Velocità       : {eps_per_sec:.1f} ep/sec")
        log(f"{'='*70}\n")

        self.last_log_time    = current_time
        self.last_log_episode = episode

    def log_new_best(self, score, episode):
        log(f"\n{'🎉'*30}")
        log(f"  🏆 NUOVO RECORD: {score} MELE! (Episodio {episode:,})")
        log(f"{'🎉'*30}\n")

    def log_checkpoint(self, episode):
        log(f"💾 Checkpoint salvato (Episodio {episode:,})")


# ══════════════════════════════════════════════════════════════
#  💾 SALVATAGGIO PERIODICO AUTOMATICO
# ══════════════════════════════════════════════════════════════

CHECKPOINT_INTERVAL = 5000
CHECKPOINT_PATH     = os.path.join(MODELS_DIR, "snake_model_checkpoint.pt")

def save_checkpoint(model, episode, best_score, epsilon):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'episode':          episode,
        'best_score':       best_score,
        'epsilon':          epsilon,
        'timestamp':        datetime.now().isoformat()
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


# ══════════════════════════════════════════════════════════════
#  🚀 SETUP INIZIALE
# ══════════════════════════════════════════════════════════════

log("\n" + "="*70)
log("🐍 SNAKE AI - TRAINING CON SUMTREE OTTIMIZZATO")
log("="*70)
log(f"  🖥️  Device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")
log(f"  🌍 Ambienti paralleli: {NUM_ENVS}")
log(f"  📦 Batch size        : {BATCH_SIZE}")
log(f"  🎯 Target episodi    : {EPISODES:,}")
log(f"  📊 Log ogni          : {LOG_INTERVAL} episodi")
log(f"  💾 Auto-save ogni    : {CHECKPOINT_INTERVAL:,} episodi")
log(f"  📝 Log file          : {LOG_FILE_PATH}")
log("="*70 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = TrainingLogger()

log("🔧 Inizializzazione modelli...")
model        = SnakeNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
target_model = SnakeNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
log("✓ Modelli creati")

best_score    = 0
start_episode = 0

if LOAD_PREVIOUS_MODEL and os.path.exists(MODEL_BEST_PATH):
    try:
        log(f"🔍 Caricamento modello da: {MODEL_BEST_PATH}")
        model.load(MODEL_BEST_PATH, device)
        log(f"✅ Modello caricato con successo!")
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH) as f:
                best_score = int(f.read().strip())
        log(f"📊 Best score precedente: {best_score}\n")
    except Exception as e:
        log(f"⚠️  Impossibile caricare: {e}")
        log("🆕 Parto da un modello nuovo.\n")
else:
    log("🆕 Nuovo training da zero.\n")

target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

log("💾 Inizializzazione SumTree Prioritized Replay Buffer...")
memory = PrioritizedReplayBuffer(
    capacity=MEMORY_SIZE,
    alpha=0.6,
    beta_start=0.4,
    beta_frames=EPISODES
)
log("✓ SumTree pronto (O(log N) sampling!)")


def select_actions_batch(states, epsilon):
    if random.random() < epsilon:
        return [random.randint(0, OUTPUT_SIZE - 1) for _ in range(len(states))]
    states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = model(states_t)
    return q_values.argmax(1).tolist()


def replay(batch_size):
    if len(memory) < batch_size:
        return

    samples, indices, weights = memory.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*samples)

    states      = torch.tensor(np.array(states),      dtype=torch.float32).to(device)
    actions     = torch.tensor(actions,                dtype=torch.long).to(device)
    rewards     = torch.tensor(rewards,                dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones       = torch.tensor(dones,                  dtype=torch.float32).to(device)
    weights     = torch.tensor(weights,                dtype=torch.float32).to(device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        best_actions  = model(next_states).argmax(1, keepdim=True)
        next_q_values = target_model(next_states).gather(1, best_actions).squeeze(1)
        target_q      = rewards + GAMMA * next_q_values * (1 - dones)

    td_errors = (q_values - target_q).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors)

    loss = (weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()


log(f"🌍 Inizializzazione {NUM_ENVS} ambienti paralleli...")
envs   = [SnakeAIEnv() for _ in range(NUM_ENVS)]
states = [env.reset() for env in envs]
log("✓ Ambienti pronti")

epsilon        = EPSILON
scores_window  = deque(maxlen=100)
total_episodes = start_episode
round_num      = 0

log("\n" + "="*70)
log("🚀 TRAINING INIZIATO!")
log("="*70)
log(f"📊 Statistiche dettagliate ogni {LOG_INTERVAL} episodi")
log(f"🎉 Ogni nuovo record verrà festeggiato!")
log(f"💾 Checkpoint automatico ogni {CHECKPOINT_INTERVAL:,} episodi")
log(f"📝 Tutto viene salvato in: {LOG_FILE_PATH}\n")

# ── training loop ─────────────────────────────────────────────
try:
    for round_num in range(EPISODES * 20):

        actions = select_actions_batch(states, epsilon)

        next_states = []
        for i, (env, action) in enumerate(zip(envs, actions)):
            next_state, reward, done = env.step(action)
            memory.push((states[i], action, reward, next_state, float(done)))

            if done:
                total_episodes += 1
                score = env.score
                scores_window.append(score)

                if score > best_score:
                    best_score = score
                    model.save(MODEL_BEST_PATH)
                    with open(BEST_SCORE_PATH, 'w') as f:
                        f.write(str(best_score))
                    logger.log_new_best(best_score, total_episodes)

                if total_episodes % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(model, total_episodes, best_score, epsilon)
                    logger.log_checkpoint(total_episodes)

                if logger.should_log(total_episodes) and len(scores_window) > 0:
                    avg = np.mean(scores_window)
                    logger.log_progress(
                        episode=total_episodes,
                        best_score=best_score,
                        avg_score=avg,
                        epsilon=epsilon,
                        memory_size=len(memory)
                    )

                next_state = env.reset()

            next_states.append(next_state)

        states = next_states

        replay(BATCH_SIZE)

        if round_num % (TARGET_UPDATE * 100) == 0:
            target_model.load_state_dict(model.state_dict())

        if round_num % 64 == 0:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if total_episodes >= EPISODES:
            break

except KeyboardInterrupt:
    log("\n\n⚠️  TRAINING INTERROTTO!")
    log("💾 Salvataggio checkpoint finale...")
    save_checkpoint(model, total_episodes, best_score, epsilon)
    log("✅ Checkpoint salvato!")

# ── fine ──────────────────────────────────────────────────────
model.save(MODEL_FINAL_PATH)

log("\n" + "="*70)
log("✅ TRAINING COMPLETATO!")
log("="*70)
log(f"  📊 Episodi totali  : {total_episodes:,}")
log(f"  🏆 Best score      : {best_score} mele")
log(f"  💾 Modello salvato : {MODEL_FINAL_PATH}")
log(f"  💾 Best model      : {MODEL_BEST_PATH}")
log(f"  💾 Checkpoint      : {CHECKPOINT_PATH}")
log(f"  📝 Log salvato in  : {LOG_FILE_PATH}")
log("="*70 + "\n")