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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from model     import SnakeNet
from snake_env import SnakeAIEnv
from config    import (
    LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN,
    BATCH_SIZE, MEMORY_SIZE, EPISODES, TARGET_UPDATE,
    MODEL_BEST_PATH, MODEL_FINAL_PATH, BEST_SCORE_PATH,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    LOAD_PREVIOUS_MODEL, NUM_ENVS, MODELS_DIR,
)

# ══════════════════════════════════════════════════════════════
#  🎨 CONFIGURAZIONE LOGGER AVANZATO
# ══════════════════════════════════════════════════════════════

class TrainingLogger:
    """Logger avanzato per il training con statistiche dettagliate"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.episodes_since_last_log = 0
        
    def log_progress(self, episode, best_score, avg_score, epsilon, beta, memory_size):
        """Log dettagliato con timing e statistiche"""
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        elapsed_since_log = current_time - self.last_log_time
        
        # Calcola velocità training
        if elapsed_since_log > 0:
            eps_per_sec = self.episodes_since_last_log / elapsed_since_log
        else:
            eps_per_sec = 0
        
        # Formatta tempo
        hours = int(elapsed_total // 3600)
        minutes = int((elapsed_total % 3600) // 60)
        seconds = int(elapsed_total % 60)
        
        print(f"\n{'='*70}")
        print(f"📊 EPISODIO {episode:,} | ⏱️  {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"{'='*70}")
        print(f"  🏆 Best Score     : {best_score:3d} mele")
        print(f"  📈 Avg (100 ep)   : {avg_score:5.2f} mele")
        print(f"  🎲 Epsilon        : {epsilon:.4f} ({(1-epsilon)*100:.1f}% sfruttamento)")
        print(f"  🎯 Beta (PER)     : {beta:.4f}")
        print(f"  💾 Memory size    : {memory_size:,} / {MEMORY_SIZE:,}")
        print(f"  ⚡ Velocità       : {eps_per_sec:.1f} ep/sec")
        print(f"{'='*70}\n")
        
        self.last_log_time = current_time
        self.episodes_since_last_log = 0
    
    def increment_episode(self):
        """Incrementa contatore episodi"""
        self.episodes_since_last_log += 1
    
    def log_new_best(self, score, episode):
        """Log quando si raggiunge un nuovo best score"""
        print(f"\n{'🎉'*30}")
        print(f"  🏆 NUOVO RECORD: {score} MELE! (Episodio {episode:,})")
        print(f"{'🎉'*30}\n")
    
    def log_checkpoint(self, episode):
        """Log quando si salva un checkpoint"""
        print(f"💾 Checkpoint salvato (Episodio {episode:,})")


# ══════════════════════════════════════════════════════════════
#  💾 SALVATAGGIO PERIODICO AUTOMATICO
# ══════════════════════════════════════════════════════════════

CHECKPOINT_INTERVAL = 5000  # Salva ogni 5000 episodi
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "snake_model_checkpoint.pt")

def save_checkpoint(model, episode, best_score, epsilon):
    """Salva checkpoint periodico"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'episode': episode,
        'best_score': best_score,
        'epsilon': epsilon,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, CHECKPOINT_PATH)


# ══════════════════════════════════════════════════════════════
#  🚀 SETUP INIZIALE
# ══════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = TrainingLogger()

print("\n" + "="*70)
print("🐍 SNAKE AI - TRAINING AVANZATO")
print("="*70)
print(f"  🖥️  Device           : {device}")
print(f"  🌍 Ambienti paralleli: {NUM_ENVS}")
print(f"  📦 Batch size        : {BATCH_SIZE}")
print(f"  🎯 Target episodi    : {EPISODES:,}")
print(f"  💾 Auto-save ogni    : {CHECKPOINT_INTERVAL:,} episodi")
print("="*70 + "\n")

# ── modelli ───────────────────────────────────────────────────
model        = SnakeNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
target_model = SnakeNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

# ── carica modello precedente ─────────────────────────────────
best_score = 0
start_episode = 0

if LOAD_PREVIOUS_MODEL and os.path.exists(MODEL_BEST_PATH):
    try:
        model.load(MODEL_BEST_PATH, device)
        print(f"✅ Modello caricato da: {MODEL_BEST_PATH}")
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH) as f:
                best_score = int(f.read().strip())
        print(f"📊 Best score precedente: {best_score}\n")
    except Exception as e:
        print(f"⚠️  Impossibile caricare: {e}\nParto da zero.\n")
else:
    print("🆕 Nuovo training da zero.\n")

target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── Prioritized Replay Memory ─────────────────────────────────
class PrioritizedMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity   = capacity
        self.alpha      = alpha
        self.memory     = []
        self.priorities = []
        self.pos        = 0

    def push(self, experience):
        max_prio = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities.append(max_prio)
        else:
            self.memory[self.pos]     = experience
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios  = np.array(self.priorities, dtype=np.float32)
        probs  = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        total   = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32).to(device)

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = float(abs(err)) + 1e-6

    def __len__(self):
        return len(self.memory)


memory = PrioritizedMemory(MEMORY_SIZE)

# ── funzioni ──────────────────────────────────────────────────
def select_actions_batch(states, epsilon):
    """Seleziona azioni per tutti gli ambienti in un solo forward pass."""
    if random.random() < epsilon:
        return [random.randint(0, OUTPUT_SIZE - 1) for _ in range(len(states))]

    states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = model(states_t)
    return q_values.argmax(1).tolist()


def replay(batch_size, beta=0.4):
    if len(memory) < batch_size:
        return

    samples, indices, weights = memory.sample(batch_size, beta)
    states, actions, rewards, next_states, dones = zip(*samples)

    states      = torch.tensor(np.array(states),      dtype=torch.float32).to(device)
    actions     = torch.tensor(actions,                dtype=torch.long).to(device)
    rewards     = torch.tensor(rewards,                dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones       = torch.tensor(dones,                  dtype=torch.float32).to(device)

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


# ── inizializza ambienti paralleli ────────────────────────────
envs   = [SnakeAIEnv() for _ in range(NUM_ENVS)]
states = [env.reset() for env in envs]

epsilon        = EPSILON
beta           = 0.4
beta_increment = (1.0 - beta) / EPISODES
scores_window  = deque(maxlen=100)
total_episodes = start_episode
round_num      = 0

print("🚀 TRAINING INIZIATO!\n")

# ── training loop ─────────────────────────────────────────────
try:
    for round_num in range(EPISODES * 20):

        # 1. Azioni batch per tutti gli ambienti
        actions = select_actions_batch(states, epsilon)

        # 2. Step in ogni ambiente
        next_states = []
        for i, (env, action) in enumerate(zip(envs, actions)):
            next_state, reward, done = env.step(action)
            memory.push((states[i], action, reward, next_state, float(done)))

            if done:
                total_episodes += 1
                logger.increment_episode()
                score = env.score
                scores_window.append(score)

                if score > best_score:
                    best_score = score
                    model.save(MODEL_BEST_PATH)
                    with open(BEST_SCORE_PATH, 'w') as f:
                        f.write(str(best_score))
                    logger.log_new_best(best_score, total_episodes)

                # 💾 SALVATAGGIO PERIODICO AUTOMATICO
                if total_episodes % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(model, total_episodes, best_score, epsilon)
                    logger.log_checkpoint(total_episodes)

                next_state = env.reset()

            next_states.append(next_state)

        states = next_states

        # 3. Training ad ogni step
        replay(BATCH_SIZE, beta)

        # 4. Target network update
        if round_num % (TARGET_UPDATE * 100) == 0:
            target_model.load_state_dict(model.state_dict())

        # 5. Epsilon decay ogni 64 step
        if round_num % 64 == 0:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            beta    = min(1.0, beta + beta_increment)

        # 6. Log dettagliato ogni 6400 step (~100 episodi)
        if round_num % 6400 == 0 and round_num > 0 and len(scores_window) > 0:
            avg = np.mean(scores_window)
            logger.log_progress(
                episode=total_episodes,
                best_score=best_score,
                avg_score=avg,
                epsilon=epsilon,
                beta=beta,
                memory_size=len(memory)
            )

        if total_episodes >= EPISODES:
            break

except KeyboardInterrupt:
    print("\n\n⚠️  TRAINING INTERROTTO!")
    print("💾 Salvataggio checkpoint finale...")
    save_checkpoint(model, total_episodes, best_score, epsilon)
    print("✅ Checkpoint salvato!")

# ── fine ──────────────────────────────────────────────────────
model.save(MODEL_FINAL_PATH)

print("\n" + "="*70)
print("✅ TRAINING COMPLETATO!")
print("="*70)
print(f"  📊 Episodi totali  : {total_episodes:,}")
print(f"  🏆 Best score      : {best_score} mele")
print(f"  💾 Modello salvato : {MODEL_FINAL_PATH}")
print(f"  💾 Best model      : {MODEL_BEST_PATH}")
print(f"  💾 Checkpoint      : {CHECKPOINT_PATH}")
print("="*70 + "\n")