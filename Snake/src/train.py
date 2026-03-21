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

from sumtree   import PrioritizedReplayBuffer
from model     import SnakeNet
from snake_env import SnakeAIEnv
from config    import (
    LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN,
    BATCH_SIZE, MEMORY_SIZE, EPISODES, TARGET_UPDATE,
    MODEL_BEST_PATH, MODEL_FINAL_PATH, BEST_SCORE_PATH,
    CNN_CHANNELS, HIDDEN_SIZE, OUTPUT_SIZE,
    LOAD_PREVIOUS_MODEL, NUM_ENVS, MODELS_DIR, LOGS_DIR,
    ROWS, COLS,
)

def print_flush(msg):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════
#  📝 LOG FILE
# ══════════════════════════════════════════════════════════════

LOG_FILE_PATH = os.path.join(LOGS_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log(msg):
    print_flush(msg)
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ══════════════════════════════════════════════════════════════
#  📊 TRAINING LOGGER
# ══════════════════════════════════════════════════════════════

LOG_INTERVAL         = 100    # report veloce ogni N episodi
EXTENDED_INTERVAL    = 500    # report diagnostico completo ogni N episodi

class TrainingLogger:
    def __init__(self):
        self.start_time       = time.time()
        self.last_log_time    = time.time()
        self.last_log_episode = 0

        # — Finestre scorrevoli ———————————————————————————
        self.scores_window     = deque(maxlen=100)   # per avg corrente
        self.scores_500        = deque(maxlen=500)   # per percentili
        self.deaths_500        = deque(maxlen=500)   # cause di morte
        self.steps_per_apple   = deque(maxlen=500)   # efficienza percorso
        self.body_at_death     = deque(maxlen=500)   # lunghezza alla morte
        self.loss_window       = deque(maxlen=200)   # loss ultime N replay
        self.grad_norm_window  = deque(maxlen=200)   # gradient norm

        # — Soglie per contatori "sopra X mele" ———————————
        self.thresholds        = [10, 15, 20, 25, 30, 35, 40]
        self.above_threshold   = {t: 0 for t in self.thresholds}

    def record_episode(self, score, death_cause, steps, apples_collected, body_len):
        """Chiamato alla fine di ogni episodio."""
        self.scores_window.append(score)
        self.scores_500.append(score)
        self.deaths_500.append(death_cause)
        self.body_at_death.append(body_len)

        if apples_collected > 0:
            self.steps_per_apple.append(steps / apples_collected)

        for t in self.thresholds:
            if score >= t:
                self.above_threshold[t] += 1

    def record_loss(self, loss_val, grad_norm_val):
        """Chiamato dopo ogni replay step."""
        self.loss_window.append(loss_val)
        self.grad_norm_window.append(grad_norm_val)

    def should_log(self, episode):
        return episode - self.last_log_episode >= LOG_INTERVAL

    def _ascii_bar(self, value, max_value, width=20, fill='█', empty='░'):
        """Genera una barra ASCII proporzionale."""
        filled = int((value / max(max_value, 1)) * width)
        return fill * filled + empty * (width - filled)

    def _elapsed(self):
        t = time.time() - self.start_time
        return int(t // 3600), int((t % 3600) // 60), int(t % 60)

    # ── REPORT VELOCE (ogni 100 episodi) ─────────────────

    def log_progress(self, episode, best_score, epsilon, memory_size):
        current_time       = time.time()
        elapsed_since_log  = current_time - self.last_log_time
        episodes_since_log = episode - self.last_log_episode
        eps_per_sec        = episodes_since_log / elapsed_since_log if elapsed_since_log > 0 else 0
        h, m, s            = self._elapsed()
        avg_score          = np.mean(self.scores_window) if self.scores_window else 0

        log(f"\n{'='*70}")
        log(f"📊 EPISODIO {episode:,} | ⏱️  {h:02d}:{m:02d}:{s:02d}")
        log(f"{'='*70}")
        log(f"  🏆 Best Score     : {best_score:3d} mele")
        log(f"  📈 Avg (100 ep)   : {avg_score:5.2f}  {self._ascii_bar(avg_score, max(best_score,1))}")
        log(f"  🎲 Epsilon        : {epsilon:.4f} ({(1-epsilon)*100:.1f}% sfruttamento)")
        log(f"  💾 Memory size    : {memory_size:,} / {MEMORY_SIZE:,}")
        log(f"  ⚡ Velocità       : {eps_per_sec:.1f} ep/sec")
        log(f"{'='*70}\n")

        self.last_log_time    = current_time
        self.last_log_episode = episode

    # ── REPORT ESTESO (ogni 500 episodi) ─────────────────

    def log_extended(self, episode, best_score, epsilon):
        if len(self.scores_500) < 10:
            return

        h, m, s   = self._elapsed()
        scores_arr = np.array(self.scores_500)

        # — Percentili ————————————————————————————————————
        p25  = np.percentile(scores_arr, 25)
        p50  = np.percentile(scores_arr, 50)
        p75  = np.percentile(scores_arr, 75)
        p95  = np.percentile(scores_arr, 95)
        mean = np.mean(scores_arr)
        std  = np.std(scores_arr)

        # — Distribuzione cause di morte ——————————————————
        deaths      = list(self.deaths_500)
        total_d     = len(deaths)
        death_types = ['wall', 'self_collision', 'starvation', 'timeout']
        death_counts = {k: deaths.count(k) for k in death_types}

        # — Efficienza ————————————————————————————————————
        avg_spa  = np.mean(self.steps_per_apple)  if self.steps_per_apple  else 0
        avg_body = np.mean(self.body_at_death)     if self.body_at_death    else 0

        # — Rete neurale ——————————————————————————————————
        avg_loss  = np.mean(self.loss_window)      if self.loss_window      else 0
        avg_grad  = np.mean(self.grad_norm_window) if self.grad_norm_window else 0
        max_grad  = np.max(self.grad_norm_window)  if self.grad_norm_window else 0

        # ── STAMPA ────────────────────────────────────────

        log(f"\n{'╔' + '═'*68 + '╗'}")
        log(f"║  🔬 REPORT ESTESO — Episodio {episode:,}  |  {h:02d}:{m:02d}:{s:02d}{' '*20}║")
        log(f"{'╠' + '═'*68 + '╣'}")

        # Score percentili
        log(f"║  📊 DISTRIBUZIONE SCORE (ultimi 500 ep)                          ║")
        log(f"║                                                                  ║")
        log(f"║   p25={p25:5.1f}  p50={p50:5.1f}  p75={p75:5.1f}  p95={p95:5.1f}              ║")
        log(f"║   media={mean:5.2f}  dev.std={std:4.2f}  best={best_score}                   ║")
        log(f"║                                                                  ║")

        # Grafico ASCII distribuzione
        log(f"║   Score  | Frequenza                                            ║")
        max_in_bucket = 1
        buckets = {}
        for sc in scores_arr:
            bucket = int(sc // 5) * 5
            buckets[bucket] = buckets.get(bucket, 0) + 1
        max_in_bucket = max(buckets.values()) if buckets else 1
        for bucket in sorted(buckets.keys()):
            count  = buckets[bucket]
            bar    = self._ascii_bar(count, max_in_bucket, width=30)
            pct    = count / total_d * 100
            log(f"║   {bucket:3d}-{bucket+4:<3d} | {bar} {pct:4.1f}%              ║")

        log(f"{'╠' + '═'*68 + '╣'}")

        # Cause di morte
        log(f"║  💀 CAUSE DI MORTE (ultimi {total_d} ep)                              ║")
        log(f"║                                                                  ║")
        for dtype in death_types:
            count = death_counts[dtype]
            pct   = count / max(total_d, 1) * 100
            bar   = self._ascii_bar(count, total_d, width=25)
            label = dtype.ljust(16)
            log(f"║   {label} {bar} {pct:5.1f}%  ({count:3d})          ║")

        log(f"{'╠' + '═'*68 + '╣'}")

        # Efficienza
        log(f"║  🐍 EFFICIENZA                                                   ║")
        log(f"║                                                                  ║")
        log(f"║   Steps medi per mela  : {avg_spa:6.1f}  (meno = più efficiente)      ║")
        log(f"║   Lunghezza media morte: {avg_body:6.1f}  segmenti                      ║")

        # Episodi sopra soglia (cumulativi)
        log(f"║                                                                  ║")
        log(f"║   Volte sopra soglia (totale):                                   ║")
        for t in self.thresholds:
            count = self.above_threshold[t]
            bar   = self._ascii_bar(count, max(self.above_threshold[self.thresholds[0]], 1), width=20)
            log(f"║     ≥{t:2d} mele: {bar} {count:5d}x              ║")

        log(f"{'╠' + '═'*68 + '╣'}")

        # Rete neurale
        log(f"║  🧠 RETE NEURALE (ultimi {len(self.loss_window)} replay)                        ║")
        log(f"║                                                                  ║")
        log(f"║   Loss media    : {avg_loss:8.5f}                                   ║")
        log(f"║   Grad norm avg : {avg_grad:8.5f}  max: {max_grad:.5f}               ║")

        # Avvisi automatici
        warnings = []
        if avg_grad > 5.0:
            warnings.append("⚠️  Gradient norm alta — possibile instabilità")
        if avg_loss > 1.0:
            warnings.append("⚠️  Loss alta — rete fatica ad apprendere")
        if death_counts['starvation'] / max(total_d, 1) > 0.3:
            warnings.append("⚠️  >30% morti per starvation — loop o apatia")
        if std > mean * 0.8:
            warnings.append("⚠️  Alta varianza score — comportamento instabile")
        if death_counts['self_collision'] / max(total_d, 1) > 0.8:
            warnings.append("ℹ️  >80% self_collision — normale a questa fase")

        if warnings:
            log(f"║                                                                  ║")
            log(f"║  AVVISI:                                                         ║")
            for w in warnings:
                log(f"║   {w:<65}║")

        log(f"{'╚' + '═'*68 + '╝'}\n")

    def log_new_best(self, score, episode):
        log(f"\n{'='*50}")
        log(f"  🏆 NUOVO RECORD: {score} MELE! (Episodio {episode:,})")
        log(f"{'='*50}\n")

    def log_checkpoint(self, episode):
        log(f"💾 Checkpoint salvato (Episodio {episode:,})")


# ══════════════════════════════════════════════════════════════
#  💾 CHECKPOINT
# ══════════════════════════════════════════════════════════════

CHECKPOINT_INTERVAL = 5000
CHECKPOINT_PATH     = os.path.join(MODELS_DIR, "snake_model_checkpoint.pt")

def save_checkpoint(model, episode, best_score, epsilon):
    torch.save({
        'model_state_dict': model.state_dict(),
        'episode':          episode,
        'best_score':       best_score,
        'epsilon':          epsilon,
        'timestamp':        datetime.now().isoformat()
    }, CHECKPOINT_PATH)


# ══════════════════════════════════════════════════════════════
#  🚀 SETUP
# ══════════════════════════════════════════════════════════════

log("\n" + "="*70)
log("🐍 SNAKE AI - CNN TRAINING")
log("="*70)
log(f"  🖥️  Device           : {'cuda' if torch.cuda.is_available() else 'cpu'}")
log(f"  🌍 Ambienti paralleli: {NUM_ENVS}")
log(f"  📦 Batch size        : {BATCH_SIZE}")
log(f"  🎯 Target episodi    : {EPISODES:,}")
log(f"  🧠 Architettura      : CNN (3x{ROWS}x{COLS}) → Dueling DQN")
log(f"  📝 Log file          : {LOG_FILE_PATH}")
log(f"  📊 Report veloce     : ogni {LOG_INTERVAL} episodi")
log(f"  🔬 Report esteso     : ogni {EXTENDED_INTERVAL} episodi")
log("="*70 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = TrainingLogger()

log("🔧 Inizializzazione modelli CNN...")
model        = SnakeNet(rows=ROWS, cols=COLS, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
target_model = SnakeNet(rows=ROWS, cols=COLS, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
log(f"✓ Modelli CNN creati — parametri: {sum(p.numel() for p in model.parameters()):,}")

best_score    = 0
start_episode = 0

if LOAD_PREVIOUS_MODEL and os.path.exists(MODEL_BEST_PATH):
    try:
        log(f"🔍 Caricamento modello da: {MODEL_BEST_PATH}")
        model.load(MODEL_BEST_PATH, device)
        log(f"✅ Modello caricato!")
        if os.path.exists(BEST_SCORE_PATH):
            with open(BEST_SCORE_PATH) as f:
                best_score = int(f.read().strip())
        log(f"📊 Best score precedente: {best_score}\n")
    except Exception as e:
        log(f"⚠️  Impossibile caricare: {e}")
        log("🆕 Parto da zero.\n")
else:
    log("🆕 Nuovo training da zero.\n")

target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

log("💾 Inizializzazione SumTree PER...")
memory = PrioritizedReplayBuffer(
    capacity=MEMORY_SIZE,
    alpha=0.6,
    beta_start=0.4,
    beta_frames=EPISODES
)
log("✓ SumTree pronto")


# ══════════════════════════════════════════════════════════════
#  🎮 FUNZIONI TRAINING
# ══════════════════════════════════════════════════════════════

def select_actions_batch(states, epsilon):
    if random.random() < epsilon:
        return [random.randint(0, OUTPUT_SIZE - 1) for _ in range(len(states))]
    states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = model(states_t)
    return q_values.argmax(1).tolist()


def replay(batch_size):
    """
    Esegue un passo di replay e restituisce (loss, grad_norm)
    per il logging diagnostico. Ritorna (None, None) se il buffer
    non è ancora abbastanza pieno.
    """
    if len(memory) < batch_size:
        return None, None

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

    # Calcola gradient norm DOPO il clip, PRIMA dello step
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()

    return loss.item(), grad_norm


def _get_death_cause(env, done):
    """
    Determina la causa di morte di un episodio.
    Chiamato subito dopo che done=True.
    """
    if not done:
        return None
    head = env.snake[0]
    if env.steps >= 2000:
        return 'timeout'
    if env.steps_since_apple > 100:
        return 'starvation'
    # Se la testa è fuori dalla griglia era un muro,
    # altrimenti era self_collision
    if not (0 <= head[0] < ROWS and 0 <= head[1] < COLS):
        return 'wall'
    return 'self_collision'


# ══════════════════════════════════════════════════════════════
#  🌍 AMBIENTI
# ══════════════════════════════════════════════════════════════

log(f"🌍 Inizializzazione {NUM_ENVS} ambienti paralleli...")
envs   = [SnakeAIEnv(ROWS, COLS) for _ in range(NUM_ENVS)]
states = [env.reset() for env in envs]
log("✓ Ambienti pronti\n")

epsilon        = EPSILON
total_episodes = start_episode

log("🚀 TRAINING INIZIATO!")
log(f"📊 Report veloce ogni {LOG_INTERVAL} ep  |  Report esteso ogni {EXTENDED_INTERVAL} ep")
log(f"📝 Log: {LOG_FILE_PATH}\n")


# ══════════════════════════════════════════════════════════════
#  🔄 TRAINING LOOP
# ══════════════════════════════════════════════════════════════

try:
    for round_num in range(EPISODES * 20):

        actions = select_actions_batch(states, epsilon)

        next_states = []
        for i, (env, action) in enumerate(zip(envs, actions)):
            next_state, reward, done = env.step(action)
            memory.push((states[i], action, reward, next_state, float(done)))

            if done:
                total_episodes += 1
                score      = env.score
                body_len   = len(env.snake)
                steps      = env.steps
                death      = _get_death_cause(env, done)

                # Registra nel logger
                logger.record_episode(
                    score        = score,
                    death_cause  = death,
                    steps        = steps,
                    apples_collected = score,
                    body_len     = body_len,
                )

                # Nuovo best
                if score > best_score:
                    best_score = score
                    model.save(MODEL_BEST_PATH)
                    with open(BEST_SCORE_PATH, 'w') as f:
                        f.write(str(best_score))
                    logger.log_new_best(best_score, total_episodes)

                # Checkpoint
                if total_episodes % CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(model, total_episodes, best_score, epsilon)
                    logger.log_checkpoint(total_episodes)

                # Report veloce
                if logger.should_log(total_episodes):
                    logger.log_progress(
                        episode      = total_episodes,
                        best_score   = best_score,
                        epsilon      = epsilon,
                        memory_size  = len(memory),
                    )

                # Report esteso
                if total_episodes % EXTENDED_INTERVAL == 0:
                    logger.log_extended(total_episodes, best_score, epsilon)

                next_state = env.reset()

            next_states.append(next_state)

        states = next_states

        # Replay + log loss/grad
        loss_val, grad_val = replay(BATCH_SIZE)
        if loss_val is not None:
            logger.record_loss(loss_val, grad_val)

        # Aggiorna target network
        if round_num % (TARGET_UPDATE * 100) == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        if round_num % 64 == 0:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if total_episodes >= EPISODES:
            break

except KeyboardInterrupt:
    log("\n⚠️  TRAINING INTERROTTO!")
    save_checkpoint(model, total_episodes, best_score, epsilon)
    log("✅ Checkpoint salvato!")
    # Stampa report esteso finale anche se interrotto
    logger.log_extended(total_episodes, best_score, epsilon)

model.save(MODEL_FINAL_PATH)

log("\n" + "="*70)
log("✅ TRAINING COMPLETATO!")
log(f"  📊 Episodi  : {total_episodes:,}")
log(f"  🏆 Best     : {best_score} mele")
log(f"  💾 Modello  : {MODEL_FINAL_PATH}")
log(f"  📝 Log      : {LOG_FILE_PATH}")
log("="*70 + "\n")