import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import sys
import numpy as np
from collections import deque

# ── path setup ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from model     import SnakeNet
from snake_env import SnakeAIEnv
from config    import (
    LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN,
    BATCH_SIZE, MEMORY_SIZE, EPISODES, TARGET_UPDATE,
    MODEL_BEST_PATH, MODEL_FINAL_PATH, BEST_SCORE_PATH,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    LOAD_PREVIOUS_MODEL,
)

# ── device ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ── ambiente e modelli ────────────────────────────────────────
env          = SnakeAIEnv()
model        = SnakeNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
target_model = SnakeNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

# ── carica modello precedente ─────────────────────────────────
best_score = 0

if LOAD_PREVIOUS_MODEL and os.path.exists(MODEL_BEST_PATH):
    try:
        model.load(MODEL_BEST_PATH, device)
        print(f"✅ Modello caricato da: {MODEL_BEST_PATH}")
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH) as f:
                best_score = int(f.read().strip())
        print(f"📊 Riprendendo da best score: {best_score}\n")
    except Exception as e:
        print(f"⚠️  Impossibile caricare il modello: {e}\nParto da zero.\n")
        best_score = 0
else:
    print("🆕 Parto da un modello nuovo.\n")

target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ── Prioritized Replay Memory ─────────────────────────────────
class PrioritizedMemory:
    """
    Replay memory con priorità: gli errori più grandi vengono
    campionati più spesso → impara più velocemente dagli errori.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity   = capacity
        self.alpha      = alpha          # quanto conta la priorità (0 = uniforme)
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
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        # Importance-sampling weights (correggono il bias)
        total  = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32).to(device)

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = float(abs(err)) + 1e-6

    def __len__(self):
        return len(self.memory)


memory = PrioritizedMemory(MEMORY_SIZE)

# ── funzioni di training ──────────────────────────────────────
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, OUTPUT_SIZE - 1)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(state_t).argmax(1).item()


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

    # Current Q
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN: scegli azione con model, valutala con target_model
    with torch.no_grad():
        best_actions  = model(next_states).argmax(1, keepdim=True)
        next_q_values = target_model(next_states).gather(1, best_actions).squeeze(1)
        target_q      = rewards + GAMMA * next_q_values * (1 - dones)

    # TD error → aggiorna priorità
    td_errors = (q_values - target_q).detach().cpu().numpy()
    memory.update_priorities(indices, td_errors)

    # Loss pesata per importance sampling
    loss = (weights * F.smooth_l1_loss(q_values, target_q, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # gradient clipping
    optimizer.step()

    return loss.item()


# ── training loop ─────────────────────────────────────────────
epsilon     = EPSILON
beta        = 0.4         # importance-sampling beta (aumenta nel tempo)
beta_max    = 1.0
beta_increment = (beta_max - beta) / EPISODES

scores_window = deque(maxlen=100)   # media mobile ultimi 100 episodi

print("=" * 60)
print("🚀 TRAINING AVVIATO")
print("=" * 60)

for episode in range(EPISODES):
    state = env.reset()
    done  = False
    episode_reward = 0

    while not done:
        action                        = select_action(state, epsilon)
        next_state, reward, done      = env.step(action)
        memory.push((state, action, reward, next_state, float(done)))
        replay(BATCH_SIZE, beta)
        episode_reward += reward
        state           = next_state

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    beta    = min(beta_max, beta + beta_increment)

    # Aggiorna target network
    if episode % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())

    score = env.score
    scores_window.append(score)
    avg_score = np.mean(scores_window)

    # Salva se nuovo best
    if score > best_score:
        best_score = score
        model.save(MODEL_BEST_PATH)
        with open(BEST_SCORE_PATH, 'w') as f:
            f.write(str(best_score))
        print(f"🏆 NEW BEST: {best_score} mele! (Episode {episode + 1})")

    if (episode + 1) % 100 == 0:
        print(f"✓ Ep {episode + 1:5d}/{EPISODES} | "
              f"Score: {score:3d} | Best: {best_score:3d} | "
              f"Avg(100): {avg_score:5.1f} | Eps: {epsilon:.3f}")

# ── fine training ─────────────────────────────────────────────
model.save(MODEL_FINAL_PATH)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETATO!")
print(f"   Best score : {best_score} mele")
print(f"   Episodi    : {EPISODES}")
print(f"   Modelli    : {MODEL_BEST_PATH}")
print("=" * 60)