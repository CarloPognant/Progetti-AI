import pygame
import torch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))

from model     import SnakeNet
from snake_env import SnakeAIEnv
from config    import (
    ROWS, COLS, CELL_SIZE, FPS,
    MODEL_BEST_PATH, VERBOSE,
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
)

# ── colori ────────────────────────────────────────────────────
BLACK       = (0,   0,   0)
GRAY        = (50,  50,  50)
DARK_GREEN  = (0,  180,  0)
BRIGHT_GREEN= (100,255, 100)
RED         = (255, 50,  50)
DARK_RED    = (200,  0,   0)
WHITE       = (255,255, 255)
YELLOW      = (255,255,   0)
CYAN        = (0,  220, 220)

# ── pygame ────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE + 80))
pygame.display.set_caption("Snake AI")
font        = pygame.font.Font(None, 36)
small_font  = pygame.font.Font(None, 24)
clock       = pygame.time.Clock()

# ── modello ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = SnakeNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    model.load(MODEL_BEST_PATH, device)
    model.eval()
    if VERBOSE:
        print(f"✓ Modello caricato: {MODEL_BEST_PATH}")
except Exception as e:
    print(f"✗ Errore nel caricamento del modello: {e}")
    pygame.quit()
    sys.exit()

env = SnakeAIEnv(ROWS, COLS)

# ── disegno ───────────────────────────────────────────────────
def draw_game(env, episode, best_score, avg_score):
    screen.fill(BLACK)

    for r in range(env.rows):
        for c in range(env.cols):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if (r + c) % 2 == 0:
                pygame.draw.rect(screen, GRAY, rect)

            if (r, c) in env.snake:
                color = BRIGHT_GREEN if (r, c) == env.snake[0] else DARK_GREEN
                pygame.draw.rect(screen, color, rect)
            elif (r, c) == env.apple:
                pygame.draw.rect(screen, RED, rect)
                pygame.draw.circle(
                    screen, DARK_RED,
                    (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2),
                    CELL_SIZE // 4
                )

            pygame.draw.rect(screen, GRAY, rect, 1)

    # HUD
    hud_y = ROWS * CELL_SIZE
    pygame.draw.rect(screen, (20, 20, 20), (0, hud_y, COLS * CELL_SIZE, 80))
    pygame.draw.line(screen, WHITE, (0, hud_y), (COLS * CELL_SIZE, hud_y), 2)

    screen.blit(font.render(f"Score: {env.score}", True, WHITE),         (10, hud_y + 8))
    screen.blit(font.render(f"Best:  {best_score}", True, YELLOW),       (10, hud_y + 42))
    screen.blit(small_font.render(f"Episodio: {episode}", True, CYAN),   (COLS * CELL_SIZE - 180, hud_y + 8))
    screen.blit(small_font.render(f"Media:  {avg_score:.1f}", True, CYAN),(COLS * CELL_SIZE - 180, hud_y + 34))

    pygame.display.flip()

# ── test loop ─────────────────────────────────────────────────
running     = True
total_score = 0
episode     = 0
best_score  = 0

if VERBOSE:
    print("\nInizio test... (ESC per uscire)\n")

while running:
    state = env.reset()
    done  = False
    episode += 1

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = done = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = done = True

        state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        with torch.no_grad():
            action = model(state_t).argmax(1).item()

        state, reward, done = env.step(action)
        avg = total_score / episode
        draw_game(env, episode, best_score, avg)
        clock.tick(FPS)

    if running:
        score        = env.score
        total_score += score
        if score > best_score:
            best_score = score
        avg = total_score / episode
        print(f"Ep {episode:4d} | Score: {score:3d} | Best: {best_score:3d} | Media: {avg:.2f}")

pygame.quit()