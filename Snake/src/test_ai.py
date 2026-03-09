import pygame
import torch
import sys
import os
import numpy as np
from model import SnakeNet
from snake_env import SnakeAIEnv
from config import (
    ROWS, COLS, CELL_SIZE, FPS,
    MODEL_BEST_PATH, VERBOSE
)

# Colori
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
DARK_GREEN = (0, 180, 0)
BRIGHT_GREEN = (100, 255, 100)
RED = (255, 50, 50)
DARK_RED = (200, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

# Pygame
pygame.init()
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE + 60))
pygame.display.set_caption("Snake AI Test")
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica modello
if VERBOSE:
    print("Caricamento modello...")
try:
    model = SnakeNet().to(device)
    model.load(MODEL_BEST_PATH, device)
    model.eval()
    if VERBOSE:
        print(f"✓ Modello caricato!")
except Exception as e:
    print(f"✗ Errore: {e}")
    pygame.quit()
    sys.exit()

env = SnakeAIEnv(ROWS, COLS)

def draw_game(env, episode):
    """Disegna il gioco"""
    screen.fill(BLACK)
    
    for r in range(env.rows):
        for c in range(env.cols):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if (r + c) % 2 == 0:
                pygame.draw.rect(screen, GRAY, rect)
            
            if (r, c) in env.snake:
                if (r, c) == env.snake[0]:
                    pygame.draw.rect(screen, BRIGHT_GREEN, rect)
                else:
                    pygame.draw.rect(screen, DARK_GREEN, rect)
            elif (r, c) == env.apple:
                pygame.draw.rect(screen, RED, rect)
                pygame.draw.circle(screen, DARK_RED,
                    (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2),
                    CELL_SIZE // 4)
            
            pygame.draw.rect(screen, GRAY, rect, 1)
    
    hud_rect = pygame.Rect(0, ROWS * CELL_SIZE, COLS * CELL_SIZE, 60)
    pygame.draw.rect(screen, DARK_GREEN, hud_rect)
    pygame.draw.line(screen, WHITE, (0, ROWS * CELL_SIZE), (COLS * CELL_SIZE, ROWS * CELL_SIZE), 2)
    
    score_text = font.render(f"Score: {env.score}", True, WHITE)
    screen.blit(score_text, (10, ROWS * CELL_SIZE + 12))
    
    episode_text = small_font.render(f"Episode: {episode}", True, YELLOW)
    screen.blit(episode_text, (COLS * CELL_SIZE - 200, ROWS * CELL_SIZE + 12))
    
    pygame.display.flip()

# Test loop
running = True
total_score = 0
episode = 0
best_score = 0

if VERBOSE:
    print("\nInizio test... (ESC per uscire)\n")

while running:
    state = env.reset()
    done = False
    episode += 1
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    done = True
        
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax(1).item()
        
        next_state, reward, done = env.step(action)
        state = next_state
        
        draw_game(env, episode)
        clock.tick(FPS)
    
    if running:
        score = env.score
        total_score += score
        if score > best_score:
            best_score = score
        
        avg_score = total_score / episode
        print(f"Episodio {episode:3d} | Score: {score:2d} | Best: {best_score:2d} | Media: {avg_score:.2f}")

if VERBOSE:
    print(f"\nTest completato! Best: {best_score}")

pygame.quit()