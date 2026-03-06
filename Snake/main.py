import pygame
import random

# --- impostazioni griglia ---
ROWS = 15
COLS = 17
CELL_SIZE = 40  # pixel per cella

# colori
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# inizializza pygame
pygame.init()
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
pygame.display.set_caption("Snake - stabile")

clock = pygame.time.Clock()

# --- inizializza snake e mela ---
snake = [(8, 4), (8, 3)]
direction = "RIGHT"
apple = (8, 13)

# --- funzione per disegnare ---
def draw_grid():
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (r, c) in snake:
                pygame.draw.rect(screen, GREEN, rect)  # snake
            elif (r, c) == apple:
                pygame.draw.rect(screen, RED, rect)    # mela
            else:
                pygame.draw.rect(screen, WHITE, rect)  # vuoto
            pygame.draw.rect(screen, BLACK, rect, 1)  # bordo cella

# --- loop principale ---
running = True
while running:
    # --- gestione eventi ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # cambia direzione solo se non è inversione
            if event.key == pygame.K_UP and direction != "DOWN":
                direction = "UP"
            elif event.key == pygame.K_DOWN and direction != "UP":
                direction = "DOWN"
            elif event.key == pygame.K_LEFT and direction != "RIGHT":
                direction = "LEFT"
            elif event.key == pygame.K_RIGHT and direction != "LEFT":
                direction = "RIGHT"

    # --- calcola nuova testa ---
    head_row, head_col = snake[0]
    if direction == "UP":
        new_head = (head_row - 1, head_col)
    elif direction == "DOWN":
        new_head = (head_row + 1, head_col)
    elif direction == "LEFT":
        new_head = (head_row, head_col - 1)
    elif direction == "RIGHT":
        new_head = (head_row, head_col + 1)

    # --- collisioni ---
    if not (0 <= new_head[0] < ROWS) or not (0 <= new_head[1] < COLS):
        print("Game Over! Hai toccato il muro.")
        running = False
    elif new_head in snake:
        print("Game Over! Hai mangiato te stesso.")
        running = False
    # --- mangia mela ---
    elif new_head == apple:
        snake.insert(0, new_head)  # cresce
        while True:
            new_apple = (random.randint(0, ROWS-1), random.randint(0, COLS-1))
            if new_apple not in snake:
                apple = new_apple
                break
    else:
        # movimento normale
        snake.insert(0, new_head)
        snake.pop()

    # --- disegna ---
    screen.fill(BLACK)
    draw_grid()
    pygame.display.flip()
    clock.tick(5)  # FPS

pygame.quit()