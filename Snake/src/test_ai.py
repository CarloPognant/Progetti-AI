import pygame
import torch
import sys
import os
import numpy as np
import json
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from model     import SnakeNet
from snake_env import SnakeAIEnv
from config    import (
    ROWS, COLS, CELL_SIZE, FPS,
    MODEL_BEST_PATH, VERBOSE,
    HIDDEN_SIZE, OUTPUT_SIZE, LOGS_DIR,
)

# ══════════════════════════════════════════════════════════════
#  🎨 PALETTE
# ══════════════════════════════════════════════════════════════

C_BG         = ( 10,  10,  20)
C_PANEL      = ( 16,  16,  32)
C_PANEL_DARK = (  8,   8,  18)
C_BORDER     = ( 45,  45,  80)
C_ACCENT     = (  0, 220, 255)
C_GOLD       = (255, 210,  50)
C_WHITE      = (230, 230, 255)
C_DIM        = (100, 100, 140)
C_RED        = (255,  65,  65)
C_GREEN_HEAD = (100, 255, 120)
C_GREEN_BODY = (  0, 180,  60)
C_APPLE      = (255,  55,  55)
C_APPLE_DARK = (180,  10,  10)
C_GRID_A     = ( 18,  18,  32)
C_GRID_B     = ( 22,  22,  40)

ACTION_LABELS  = ["◄  Sinistra", "^  Avanti", "►  Destra"]
ACTION_BAR_COL = [(255, 90, 90), (80, 240, 80), (80, 160, 255)]

# ══════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════

CELL    = CELL_SIZE
GAME_W  = COLS * CELL
GAME_H  = ROWS * CELL
HUD_H   = 64
PANEL_W = 310
TOTAL_W = GAME_W + PANEL_W
TOTAL_H = GAME_H + HUD_H

# ══════════════════════════════════════════════════════════════
#  PYGAME INIT
# ══════════════════════════════════════════════════════════════

pygame.init()
screen = pygame.display.set_mode((TOTAL_W, TOTAL_H))
pygame.display.set_caption("Snake AI  -  Debug View")
clock  = pygame.time.Clock()

F_BIG   = pygame.font.SysFont("consolas", 22, bold=True)
F_MED   = pygame.font.SysFont("consolas", 17)
F_SMALL = pygame.font.SysFont("consolas", 14)
F_TINY  = pygame.font.SysFont("consolas", 12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════
#  MODELLO
# ══════════════════════════════════════════════════════════════

try:
    model = SnakeNet(rows=ROWS, cols=COLS,
                     hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
except TypeError:
    model = SnakeNet(hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)

try:
    model.load(MODEL_BEST_PATH, device)
    model.eval()
    if VERBOSE:
        print(f"✓ Modello caricato: {MODEL_BEST_PATH}")
except Exception as e:
    print(f"✗ Errore caricamento modello: {e}")
    pygame.quit()
    sys.exit()

env = SnakeAIEnv(ROWS, COLS)

# ══════════════════════════════════════════════════════════════
#  STATO RUNTIME
# ══════════════════════════════════════════════════════════════

show_saliency = False
show_channels = False
q_smooth      = np.zeros(OUTPUT_SIZE, dtype=np.float32)
score_history = deque(maxlen=24)

# ══════════════════════════════════════════════════════════════
#  📝 MOVE LOGGER
# ══════════════════════════════════════════════════════════════

ACTION_NAMES_LOG = ["sinistra", "avanti", "destra"]

class MoveLogger:
    """
    Registra ogni mossa dell'AI con contesto completo.
    Salva un file JSON per episodio nella cartella logs/.

    Struttura del report:
      episode_info  — numero, score, causa morte, durata
      moves[]       — ogni step: posizioni, Q-values, azione, risultato
      analysis      — pattern automatici rilevati (trappole, loop, greedy)
    """

    def __init__(self):
        self.recording  = False
        self.moves      = []
        self.ep_start   = None
        self._prev_state = None

    def start_episode(self):
        if not self.recording:
            return
        self.moves    = []
        self.ep_start = datetime.now()

    def log_step(self, step, env_obj, q_vals, action, reward, done):
        """Registra un singolo step con tutte le info rilevanti."""
        if not self.recording:
            return

        head = env_obj.snake[0]
        apple = env_obj.apple
        dist  = abs(head[0] - apple[0]) + abs(head[1] - apple[1])

        # Calcola spazio libero (flood fill già disponibile nell'env)
        free_space = env_obj._flood_fill(head)

        # Margine di confidenza: gap tra 1° e 2° Q-value
        sorted_q   = np.sort(q_vals)[::-1]
        q_margin   = float(sorted_q[0] - sorted_q[1])

        # Classifica il risultato di questa mossa
        if done:
            if env_obj.steps >= 2000:
                result = "timeout"
            elif env_obj.steps_since_apple > 100:
                result = "starvation"
            elif not (0 <= head[0] < ROWS and 0 <= head[1] < COLS):
                result = "wall"
            else:
                result = "self_collision"
        elif reward >= 15:
            result = "apple"
        else:
            result = "move"

        self.moves.append({
            "step":        step,
            "head":        list(head),
            "apple":       list(apple),
            "dist_apple":  dist,
            "body_len":    len(env_obj.snake),
            "free_space":  free_space,
            "direction":   list(env_obj.direction),
            "q_left":      round(float(q_vals[0]), 4),
            "q_forward":   round(float(q_vals[1]), 4),
            "q_right":     round(float(q_vals[2]), 4),
            "action":      ACTION_NAMES_LOG[action],
            "q_margin":    round(q_margin, 4),
            "reward":      round(float(reward), 3),
            "result":      result,
            "steps_no_apple": env_obj.steps_since_apple,
            "visit_count": env_obj.visited_positions.get(head, 0),
        })

    def save_episode(self, episode, score, best_score):
        """Analizza le mosse e salva il report JSON."""
        if not self.recording or not self.moves:
            return None

        # ── Analisi automatica ────────────────────────────────
        traps        = [m for m in self.moves if m["free_space"] < m["body_len"] * 1.5]
        loops        = [m for m in self.moves if m["visit_count"] >= 3]
        low_conf     = [m for m in self.moves if m["q_margin"] < 0.5]
        death_move   = self.moves[-1] if self.moves else None
        apple_steps  = [m for m in self.moves if m["result"] == "apple"]

        # Step prima di ogni mela: misura efficienza
        apple_efficiency = []
        last_apple_step  = 0
        for m in self.moves:
            if m["result"] == "apple":
                apple_efficiency.append(m["step"] - last_apple_step)
                last_apple_step = m["step"]

        analysis = {
            "death_cause":       death_move["result"] if death_move else "unknown",
            "death_step":        death_move["step"]   if death_move else 0,
            "death_head":        death_move["head"]   if death_move else [],
            "death_body_len":    death_move["body_len"] if death_move else 0,
            "death_free_space":  death_move["free_space"] if death_move else 0,
            "death_q_margin":    death_move["q_margin"] if death_move else 0,
            "trap_entries":      len(traps),
            "trap_steps":        [m["step"] for m in traps[:10]],
            "loop_entries":      len(loops),
            "low_confidence_moves": len(low_conf),
            "avg_q_margin":      round(float(np.mean([m["q_margin"] for m in self.moves])), 4),
            "avg_free_space":    round(float(np.mean([m["free_space"] for m in self.moves])), 2),
            "apples_collected":  score,
            "steps_per_apple":   apple_efficiency,
            "avg_steps_per_apple": round(float(np.mean(apple_efficiency)), 1) if apple_efficiency else 0,
        }

        if self.ep_start is None:
            self.ep_start = datetime.now()
        duration = (datetime.now() - self.ep_start).total_seconds()

        report = {
            "episode":    episode,
            "score":      score,
            "best_score": best_score,
            "total_steps": len(self.moves),
            "duration_sec": round(duration, 2),
            "grid":       f"{ROWS}x{COLS}",
            "timestamp":  datetime.now().isoformat(),
            "analysis":   analysis,
            "moves":      self.moves,
        }

        # Salva file
        os.makedirs(LOGS_DIR, exist_ok=True)
        fname    = f"ep{episode:04d}_score{score}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = os.path.join(LOGS_DIR, fname)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"  📝 Report salvato: {filepath}")
        print(f"     Causa morte : {analysis['death_cause']}"
              f"  |  Trappole  : {analysis['trap_entries']}"
              f"  |  Loop      : {analysis['loop_entries']}")
        return filepath

    def toggle(self):
        self.recording = not self.recording
        print(f"  Registrazione mosse: {'ON' if self.recording else 'OFF'}")

move_logger = MoveLogger()

# ══════════════════════════════════════════════════════════════
#  ANALISI
# ══════════════════════════════════════════════════════════════

def get_q_values(state_np):
    state_t = torch.tensor(np.array([state_np]), dtype=torch.float32).to(device)
    with torch.no_grad():
        q = model(state_t)
    return q[0].cpu().numpy()


def compute_saliency(state_np):
    if state_np.ndim != 3:
        return None
    try:
        state_t = torch.tensor(np.array([state_np]),
                               dtype=torch.float32).to(device)
        state_t.requires_grad_(True)
        q    = model(state_t)
        best = int(q.argmax(1).item())
        model.zero_grad()
        q[0][best].backward()
        if state_t.grad is None:
            return None
        g = state_t.grad.data.abs()
        if g.ndim == 4:
            sal = g[0].sum(dim=0).cpu().numpy()
        elif g.ndim == 3:
            sal = g.sum(dim=0).cpu().numpy()
        else:
            return None
        if sal.shape != (ROWS, COLS):
            return None
        if sal.max() > 1e-8:
            sal = sal / sal.max()
        return sal.astype(np.float32)
    except Exception as e:
        print(f"  [saliency] {e}")
        return None

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_text(surf, text, font, color, x, y, align="left"):
    s = font.render(text, True, color)
    if align == "center":
        x -= s.get_width() // 2
    elif align == "right":
        x -= s.get_width()
    surf.blit(s, (x, y))
    return s.get_height()


def draw_sep(y_pos, x_start=None, width=None):
    x0 = x_start if x_start is not None else GAME_W + 14
    w  = width   if width   is not None else PANEL_W - 28
    pygame.draw.line(screen, C_BORDER, (x0, y_pos), (x0 + w, y_pos), 1)


# ══════════════════════════════════════════════════════════════
#  DRAW: GRIGLIA
# ══════════════════════════════════════════════════════════════

def draw_grid():
    snake_set = set(env.snake)
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
            pygame.draw.rect(screen, C_GRID_A if (r + c) % 2 == 0 else C_GRID_B, rect)

            if (r, c) in snake_set:
                if (r, c) == env.snake[0]:
                    pygame.draw.rect(screen, C_GREEN_HEAD, rect, border_radius=5)
                    ex = c * CELL + CELL // 2
                    ey = r * CELL + CELL // 2
                    pygame.draw.circle(screen, (10, 20, 10), (ex - 5, ey - 4), 3)
                    pygame.draw.circle(screen, (10, 20, 10), (ex + 5, ey - 4), 3)
                    pygame.draw.circle(screen, C_ACCENT,    (ex - 5, ey - 4), 1)
                    pygame.draw.circle(screen, C_ACCENT,    (ex + 5, ey - 4), 1)
                else:
                    idx   = env.snake.index((r, c))
                    ratio = 1.0 - idx / max(len(env.snake), 1) * 0.6
                    col   = tuple(int(C_GREEN_BODY[i] * ratio) for i in range(3))
                    pygame.draw.rect(screen, col, rect.inflate(-2, -2), border_radius=4)

            elif (r, c) == env.apple:
                cx = c * CELL + CELL // 2
                cy = r * CELL + CELL // 2
                pygame.draw.circle(screen, C_APPLE,      (cx, cy), CELL // 2 - 3)
                pygame.draw.circle(screen, C_APPLE_DARK, (cx, cy), CELL // 4)
                pygame.draw.line(screen, (80, 200, 80),
                                 (cx, cy - CELL // 2 + 3),
                                 (cx + 4, cy - CELL // 2 - 2), 2)


# ══════════════════════════════════════════════════════════════
#  DRAW: SALIENCY OVERLAY
# ══════════════════════════════════════════════════════════════

def draw_saliency(state_np):
    sal = compute_saliency(state_np)
    if sal is None:
        return
    overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
    for r in range(ROWS):
        for c in range(COLS):
            v = float(sal[r, c])
            if v < 0.05:
                continue
            if v < 0.5:
                col = lerp_color((80, 0, 180), (255, 60, 0), v * 2)
            else:
                col = lerp_color((255, 60, 0), (255, 240, 0), (v - 0.5) * 2)
            overlay.fill((*col, int(v * 180)),
                         rect=pygame.Rect(c * CELL, r * CELL, CELL, CELL))
    screen.blit(overlay, (0, 0))


# ══════════════════════════════════════════════════════════════
#  DRAW: CANALI CNN
# ══════════════════════════════════════════════════════════════

def draw_channels(state_np, x, y):
    if state_np.ndim != 3 or state_np.shape[0] < 3:
        draw_text(screen, "Stato non-CNN (flat)", F_SMALL, C_RED, x, y)
        return
    names  = ["Corpo", "Testa", "Mela"]
    tints  = [(40, 220, 100), (140, 255, 140), (255, 70, 70)]
    cs     = 4
    ch_w   = COLS * cs + 6
    draw_text(screen, "Input CNN (3 canali):", F_SMALL, C_GOLD, x, y)
    y += 16
    for ch in range(3):
        cx = x + ch * ch_w
        draw_text(screen, names[ch], F_TINY, tints[ch], cx, y)
        cy = y + 13
        pygame.draw.rect(screen, C_BORDER,
                         pygame.Rect(cx - 1, cy - 1,
                                     COLS * cs + 2, ROWS * cs + 2), 1)
        for r in range(ROWS):
            for c2 in range(COLS):
                v   = float(state_np[ch, r, c2])
                col = tuple(int(tints[ch][i] * max(v, 0.08)) for i in range(3)) \
                      if v > 0 else (14, 14, 28)
                pygame.draw.rect(screen, col,
                                 (cx + c2 * cs, cy + r * cs, cs, cs))


# ══════════════════════════════════════════════════════════════
#  DRAW: HUD INFERIORE
# ══════════════════════════════════════════════════════════════

def draw_hud(episode, best_score, avg):
    hy = GAME_H
    pygame.draw.rect(screen, C_PANEL_DARK, (0, hy, GAME_W, HUD_H))
    pygame.draw.line(screen, C_BORDER, (0, hy), (GAME_W, hy), 1)

    draw_text(screen, f"SCORE  {env.score:3d}", F_BIG, C_WHITE,  14, hy + 8)
    draw_text(screen, f"BEST   {best_score:3d}", F_BIG, C_GOLD,  14, hy + 34)
    draw_text(screen, f"EP {episode}",  F_MED, C_ACCENT, GAME_W - 14, hy + 8,  align="right")
    draw_text(screen, f"AVG {avg:.1f}", F_MED, C_DIM,    GAME_W - 14, hy + 32, align="right")

    # Indicatore REC
    if move_logger.recording:
        rec_x = GAME_W // 2 + 110
        pygame.draw.circle(screen, C_RED, (rec_x, hy + 16), 6)
        draw_text(screen, "REC", F_SMALL, C_RED, rec_x + 12, hy + 9)

    # barra steps-since-apple
    ratio  = min(env.steps_since_apple / 100, 1.0)
    bx, bw, bh, by2 = GAME_W // 2 - 100, 200, 8, hy + HUD_H // 2 - 4
    pygame.draw.rect(screen, C_PANEL, (bx, by2, bw, bh), border_radius=4)
    if ratio > 0:
        pygame.draw.rect(screen, lerp_color(C_ACCENT, C_RED, ratio),
                         (bx, by2, int(ratio * bw), bh), border_radius=4)
    pygame.draw.rect(screen, C_BORDER, (bx, by2, bw, bh), 1, border_radius=4)
    draw_text(screen, "passi senza mela", F_TINY, C_DIM,
              bx + bw // 2, by2 - 14, align="center")


# ══════════════════════════════════════════════════════════════
#  DRAW: PANNELLO LATERALE
# ══════════════════════════════════════════════════════════════

def draw_panel(q_vals, episode, best_score, avg, state_np):
    px  = GAME_W
    pw  = PANEL_W
    pad = 14
    pygame.draw.rect(screen, C_PANEL, (px, 0, pw, TOTAL_H))
    pygame.draw.line(screen, C_BORDER, (px, 0), (px, TOTAL_H), 2)

    y = 14

    # Titolo
    draw_text(screen, "SNAKE AI  DEBUG", F_BIG, C_ACCENT, px + pad, y)
    y += 30
    draw_sep(y)
    y += 10

    # Q-Values
    draw_text(screen, "Q-VALUES", F_SMALL, C_GOLD, px + pad, y)
    y += 18

    best_a  = int(np.argmax(q_vals))
    q_min   = float(q_vals.min())
    q_range = max(float(q_vals.max()) - q_min, 1e-4)
    bw      = pw - pad * 2 - 6

    for i in range(OUTPUT_SIZE):
        is_b   = (i == best_a)
        fill   = max(int(((q_vals[i] - q_min) / q_range) * bw), 2)
        bg_r   = pygame.Rect(px + pad, y, bw, 24)
        fil_r  = pygame.Rect(px + pad, y, fill, 24)
        fc     = ACTION_BAR_COL[i] if is_b else tuple(c // 3 for c in ACTION_BAR_COL[i])

        pygame.draw.rect(screen, C_PANEL_DARK, bg_r,  border_radius=5)
        pygame.draw.rect(screen, fc,           fil_r, border_radius=5)
        if is_b:
            pygame.draw.rect(screen, C_WHITE, bg_r, 1, border_radius=5)

        tc = C_WHITE if is_b else C_DIM
        draw_text(screen, ACTION_LABELS[i],      F_SMALL, tc, px + pad + 5, y + 5)
        draw_text(screen, f"{q_vals[i]:+.2f}",   F_SMALL, tc, px + pw - pad - 4, y + 5, align="right")
        if is_b:
            draw_text(screen, "*", F_SMALL, C_GOLD, px + pw - pad - 30, y + 5)
        y += 30

    y += 4
    draw_sep(y)
    y += 10

    # Stats
    draw_text(screen, "STATISTICHE", F_SMALL, C_GOLD, px + pad, y)
    y += 18

    def srow(label, val, col=C_WHITE):
        nonlocal y
        draw_text(screen, label, F_SMALL, C_DIM,   px + pad,      y)
        draw_text(screen, str(val), F_SMALL, col, px + pw - pad, y, align="right")
        y += 18

    srow("Episodio",     episode)
    srow("Score",        env.score,                     C_ACCENT)
    srow("Best",         f"{best_score} mele",          C_GOLD)
    srow("Media",        f"{avg:.1f}")
    srow("Lunghezza",    f"{len(env.snake)} seg")
    srow("Steps",        env.steps)
    srow("Steps/mela",   env.steps_since_apple,
         C_RED if env.steps_since_apple > 60 else C_WHITE)

    y += 4
    draw_sep(y)
    y += 10

    # Mini grafico storico
    if len(score_history) > 1:
        draw_text(screen, "STORICO PARTITE", F_SMALL, C_GOLD, px + pad, y)
        y += 16
        gh  = 50
        gw  = pw - pad * 2
        gr  = pygame.Rect(px + pad, y, gw, gh)
        pygame.draw.rect(screen, C_PANEL_DARK, gr, border_radius=4)
        pygame.draw.rect(screen, C_BORDER,     gr, 1, border_radius=4)
        sl  = list(score_history)
        mx  = max(max(sl), 1)
        bw2 = gw // len(sl)
        for idx, sc in enumerate(sl):
            bh2 = int((sc / mx) * (gh - 6))
            bx2 = px + pad + idx * bw2 + 1
            by2 = y + gh - bh2 - 2
            if bh2 > 0:
                col = lerp_color(C_GREEN_BODY, C_GOLD, sc / mx)
                pygame.draw.rect(screen, col,
                                 pygame.Rect(bx2, by2, max(bw2 - 2, 1), bh2),
                                 border_radius=2)
        # linea best
        by_best = y + gh - int((best_score / mx) * (gh - 6)) - 2
        if y <= by_best <= y + gh:
            pygame.draw.line(screen, C_GOLD,
                             (px + pad, by_best), (px + pad + gw, by_best), 1)
        y += gh + 8

    draw_sep(y)
    y += 10

    # Controlli
    draw_text(screen, "CONTROLLI", F_SMALL, C_GOLD, px + pad, y)
    y += 18

    def trow(key, label, on):
        nonlocal y
        sc = (80, 255, 100) if on else C_DIM
        ss = "ON " if on else "OFF"
        draw_text(screen, f"[{key}]", F_SMALL, C_ACCENT, px + pad,      y)
        draw_text(screen, label,      F_SMALL, C_WHITE,  px + pad + 36, y)
        draw_text(screen, ss,         F_SMALL, sc, px + pw - pad, y, align="right")
        y += 18

    trow("S", "Saliency map", show_saliency)
    trow("C", "Canali CNN",   show_channels)
    trow("R", "Registra",     move_logger.recording)
    draw_text(screen, "[ESC]  Esci", F_SMALL, C_RED, px + pad, y)
    y += 24

    # Canali CNN
    if show_channels:
        draw_sep(y)
        y += 8
        draw_channels(state_np, px + pad, y)


# ══════════════════════════════════════════════════════════════
#  GAME LOOP
# ══════════════════════════════════════════════════════════════

running     = True
total_score = 0
episode     = 0
best_score  = 0

if VERBOSE:
    print("\n  [S] saliency  [C] canali CNN  [R] registra mosse  [ESC] esci\n")

while running:
    state = env.reset()
    done  = False
    episode += 1
    step  = 0
    move_logger.start_episode()

    while not done and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = done = True
                elif event.key == pygame.K_s:
                    show_saliency = not show_saliency
                    print(f"  Saliency: {'ON' if show_saliency else 'OFF'}")
                elif event.key == pygame.K_c:
                    show_channels = not show_channels
                    print(f"  Canali CNN: {'ON' if show_channels else 'OFF'}")
                elif event.key == pygame.K_r:
                    move_logger.toggle()

        q_values  = get_q_values(state)
        q_smooth  = q_smooth * 0.6 + q_values * 0.4
        action    = int(np.argmax(q_values))

        state, reward, done = env.step(action)
        step += 1

        # Log DOPO lo step così abbiamo reward e done aggiornati
        move_logger.log_step(step, env, q_values, action, reward, done)

        screen.fill(C_BG)
        draw_grid()
        if show_saliency:
            draw_saliency(state)
        avg = total_score / max(episode, 1)
        draw_hud(episode, best_score, avg)
        draw_panel(q_smooth, episode, best_score, avg, state)
        pygame.display.flip()
        clock.tick(FPS)

    if running:
        score        = env.score
        total_score += score
        best_score   = max(best_score, score)
        score_history.append(score)
        avg          = total_score / episode
        move_logger.save_episode(episode, score, best_score)
        print(f"Ep {episode:4d} | Score {score:3d} | Best {best_score:3d} | Media {avg:.2f}")

pygame.quit()