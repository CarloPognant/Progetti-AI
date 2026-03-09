import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from collections import deque
from model import SnakeNet
from snake_env import SnakeAIEnv

# Iperparametri OTTIMIZZATI
LEARNING_RATE = 0.0005  # ↓ Più basso per stabilità
GAMMA = 0.99  # fattore di sconto
EPSILON = 1.0  # tasso di esplorazione
EPSILON_DECAY = 0.9995  # ↓ Scende più lentamente (esplora più a lungo!)
EPSILON_MIN = 0.05 # ↓ Permette un po' di esplorazione anche alla fine
BATCH_SIZE = 64  # ↑ Aumentato
MEMORY_SIZE = 50000  # ↑ Aumentato da 10000
EPISODES = 25000  # ↑ Aumentato da 500
TARGET_UPDATE = 25  # ↓ Aggiorna più spesso (da 50)

# Device (GPU se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Inizializzazione
env = SnakeAIEnv()
model = SnakeNet().to(device)
target_model = SnakeNet().to(device)

# 🔄 CONTINUA DA MODELLO PRECEDENTE O RIPETI DA ZERO?
LOAD_PREVIOUS_MODEL = True  # ← Metti False per ripartire da zero
BEST_SCORE_FROM_PREVIOUS = 30  # ← CAMBIA QUESTO SE CONTINUI! Metti il valore corretto (30, 40, ecc)

# DEBUG: Vedi dove sta cercando il file
current_dir = os.getcwd()
model_path = os.path.join(current_dir, "snake_model_best.pt")
print(f"\n📁 Directory corrente: {current_dir}")
print(f"📄 Cerco il modello in: {model_path}")
print(f"✓ File esiste? {os.path.exists(model_path)}\n")

best_score = 0  # Iniziale

if LOAD_PREVIOUS_MODEL and os.path.exists("snake_model_best.pt"):
    try:
        model.load("snake_model_best.pt")
        print("✅ Modello precedente caricato! Continuo l'addestramento...")
        # IMPORTANTE: Quando continui il training, IMPOSTA best_score al valore corretto!
        best_score = BEST_SCORE_FROM_PREVIOUS
        print(f"📊 Riprendendo da Best Score: {best_score} mele\n")
    except Exception as e:
        print(f"⚠️  Impossibile caricare il modello: {e}")
        print("Parto da zero.\n")
        best_score = 0
else:
    if LOAD_PREVIOUS_MODEL:
        print("🆕 File non trovato! Parto da un modello nuovo (zero)\n")
    else:
        print("🆕 LOAD_PREVIOUS_MODEL = False. Parto da un modello nuovo (zero)\n")
    best_score = 0

target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

def remember(state, action, reward, next_state, done):
    """Salva l'esperienza nella replay memory"""
    memory.append((state, action, reward, next_state, done))

def replay(batch_size):
    """Allena il modello su un batch di esperienze"""
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    
    # Predizioni attuali
    q_values = model(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q-values (con target network)
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    
    # Loss e backprop
    loss = F.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def select_action(state, epsilon):
    """Epsilon-greedy action selection"""
    if random.random() < epsilon:
        return random.randint(0, 2)  # azione casuale
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state)
            return q_values.argmax(1).item()

# Training loop
epsilon = EPSILON

for episode in range(EPISODES):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        
        remember(state, action, reward, next_state, done)
        replay(BATCH_SIZE)
        
        episode_reward += reward
        state = next_state
    
    # Decrement epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_model.load_state_dict(model.state_dict())
    
    # Stampa progresso
    score = env.score
    if score > best_score:
        best_score = score
        model.save(f"snake_model_best.pt")
        print(f"🏆 NEW BEST: {best_score} mele! (Episode {episode + 1})")
    
    if (episode + 1) % 50 == 0:
        print(f"✓ Episode {episode + 1}/{EPISODES} | Score: {score} | Best: {best_score} | Epsilon: {epsilon:.3f}")

# Salva il modello finale
model.save("snake_model_final.pt")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETATO!")
print("=" * 60)
print(f"📈 Risultati finali:")
print(f"   - Episodi: {episode + 1}")
print(f"   - Best score: {best_score} mele")
print(f"\n💾 Modelli salvati:")
print(f"   - snake_model_best.pt  (miglior modello)")
print(f"   - snake_model_final.pt (modello finale)")
print(f"\n🎮 Prossimo step:")
print(f"   python test_ai.py")
print("=" * 60)