import pygame
import random
import sys

# --- impostazioni griglia ---
ROWS = 15
COLS = 17
CELL_SIZE = 40  # pixel per cella

# colori
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (0, 180, 0)
BRIGHT_GREEN = (100, 255, 100)
RED = (255, 50, 50)
DARK_RED = (200, 0, 0)
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)

# inizializza pygame
pygame.init()
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE + 60))
pygame.display.set_caption("Snake Game")
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

clock = pygame.time.Clock()
FPS = 10  # Velocità del gioco (più veloce del codice originale)

class SnakeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Resetta il gioco"""
        self.snake = [(ROWS // 2, COLS // 2), (ROWS // 2, COLS // 2 - 1)]
        self.direction = "RIGHT"
        self.next_direction = "RIGHT"
        self.apple = self._spawn_apple()
        self.score = 0
        self.game_over = False
        self.game_over_reason = ""
    
    def _spawn_apple(self):
        """Genera una mela in posizione casuale"""
        while True:
            apple = (random.randint(0, ROWS - 1), random.randint(0, COLS - 1))
            if apple not in self.snake:
                return apple
    
    def handle_input(self):
        """Gestisce l'input da tastiera"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != "DOWN":
                    self.next_direction = "UP"
                elif event.key == pygame.K_DOWN and self.direction != "UP":
                    self.next_direction = "DOWN"
                elif event.key == pygame.K_LEFT and self.direction != "RIGHT":
                    self.next_direction = "LEFT"
                elif event.key == pygame.K_RIGHT and self.direction != "LEFT":
                    self.next_direction = "RIGHT"
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def update(self):
        """Aggiorna lo stato del gioco"""
        if self.game_over:
            return
        
        self.direction = self.next_direction
        
        # Calcola nuova testa
        head_row, head_col = self.snake[0]
        if self.direction == "UP":
            new_head = (head_row - 1, head_col)
        elif self.direction == "DOWN":
            new_head = (head_row + 1, head_col)
        elif self.direction == "LEFT":
            new_head = (head_row, head_col - 1)
        elif self.direction == "RIGHT":
            new_head = (head_row, head_col + 1)
        
        # Verifica collisioni
        if not (0 <= new_head[0] < ROWS) or not (0 <= new_head[1] < COLS):
            self.game_over = True
            self.game_over_reason = "Hai toccato il muro!"
        elif new_head in self.snake:
            self.game_over = True
            self.game_over_reason = "Hai mangiato te stesso!"
        else:
            self.snake.insert(0, new_head)
            
            # Verifica se ha mangiato la mela
            if new_head == self.apple:
                self.score += 1
                self.apple = self._spawn_apple()
            else:
                self.snake.pop()
    
    def draw(self):
        """Disegna il gioco"""
        screen.fill(BLACK)
        
        # Disegna la griglia
        for r in range(ROWS):
            for c in range(COLS):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Sfondo cella
                if (r + c) % 2 == 0:
                    pygame.draw.rect(screen, GRAY, rect)
                
                # Serpente
                if (r, c) in self.snake:
                    if (r, c) == self.snake[0]:  # Testa
                        pygame.draw.rect(screen, BRIGHT_GREEN, rect)
                    else:
                        pygame.draw.rect(screen, DARK_GREEN, rect)
                # Mela
                elif (r, c) == self.apple:
                    pygame.draw.rect(screen, RED, rect)
                    pygame.draw.circle(
                        screen,
                        DARK_RED,
                        (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2),
                        CELL_SIZE // 4
                    )
                
                # Bordo cella
                pygame.draw.rect(screen, GRAY, rect, 1)
        
        # Disegna HUD
        hud_rect = pygame.Rect(0, ROWS * CELL_SIZE, COLS * CELL_SIZE, 60)
        pygame.draw.rect(screen, DARK_GREEN, hud_rect)
        pygame.draw.line(screen, WHITE, (0, ROWS * CELL_SIZE), (COLS * CELL_SIZE, ROWS * CELL_SIZE), 2)
        
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, ROWS * CELL_SIZE + 12))
        
        if self.game_over:
            game_over_text = font.render(f"GAME OVER! {self.game_over_reason}", True, RED)
            screen.blit(game_over_text, (COLS * CELL_SIZE // 2 - game_over_text.get_width() // 2, ROWS * CELL_SIZE + 12))
            
            restart_text = small_font.render("Premi R per ricominciare, ESC per uscire", True, YELLOW)
            screen.blit(restart_text, (COLS * CELL_SIZE // 2 - restart_text.get_width() // 2, ROWS * CELL_SIZE + 40))
        
        pygame.display.flip()

# --- Main loop ---
game = SnakeGame()
running = True

while running:
    running = game.handle_input()
    game.update()
    game.draw()
    clock.tick(FPS)

pygame.quit()
sys.exit()
