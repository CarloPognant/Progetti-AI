import random
import numpy as np

class SnakeAIEnv:
    """
    Ambiente per allenare l'AI del Snake.
    Lo stato è un array di 11 valori che rappresentano:
    - Direzione corrente (4 valori one-hot)
    - Posizione della mela relativa al serpente (2 valori)
    - Pericoli intorno (up, down, left, right) = 4 valori
    - 1 valore di riempimento per arrivare a 11
    """
    
    def __init__(self, rows=15, cols=17):
        self.rows = rows
        self.cols = cols
        self.reset()
    
    def reset(self):
        """Resetta il gioco e ritorna lo stato iniziale"""
        self.snake = [(self.rows // 2, self.cols // 2)]
        self.direction = (0, 1)  # (row_delta, col_delta)
        self.apple = self._spawn_apple()
        self.steps = 0
        self.score = 0
        return self._get_state()
    
    def _spawn_apple(self):
        """Genera una nuova mela in una posizione casuale"""
        while True:
            apple = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
            if apple not in self.snake:
                return apple
    
    def step(self, action):
        """
        Esegue un passo del gioco.
        action: 0 = gira a sinistra, 1 = dritto, 2 = gira a destra
        ritorna: (state, reward, done)
        """
        # Cambia direzione in base all'azione
        direction_map = {
            (0, 1): [(-1, 0), (0, 1), (1, 0)],    # RIGHT: up, straight, down
            (0, -1): [(1, 0), (0, -1), (-1, 0)],   # LEFT: down, straight, up
            (1, 0): [(0, 1), (1, 0), (0, -1)],     # DOWN: right, straight, left
            (-1, 0): [(0, -1), (-1, 0), (0, 1)]    # UP: left, straight, right
        }
        
        self.direction = direction_map[self.direction][action]
        
        # Calcola la nuova testa
        head_row, head_col = self.snake[0]
        new_head = (
            head_row + self.direction[0],
            head_col + self.direction[1]
        )
        
        # Verifica collisioni
        reward = 0.1  # reward piccolo per ogni passo sopravvissuto
        done = False
        
        # Collisione con il muro
        if not (0 <= new_head[0] < self.rows) or not (0 <= new_head[1] < self.cols):
            reward = -1
            done = True
        # Collisione con se stesso
        elif new_head in self.snake:
            reward = -1
            done = True
        # Mangia la mela
        elif new_head == self.apple:
            self.snake.insert(0, new_head)
            self.apple = self._spawn_apple()
            reward = 10  # grande reward per aver mangiato
            self.score += 1
        else:
            # Movimento normale
            self.snake.insert(0, new_head)
            self.snake.pop()
        
        self.steps += 1
        if self.steps > 500:  # Limite di passi per episodio (aumentato per più tempo)
            done = True
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        """
        Ritorna lo stato come array numpy di 11 elementi:
        - [0-3] Direzione one-hot (UP, DOWN, LEFT, RIGHT)
        - [4-5] Posizione mela relativa (delta_row, delta_col)
        - [6-9] Pericoli intorno (UP, DOWN, LEFT, RIGHT)
        - [10] Lunghezza serpente normalizzata
        """
        head_row, head_col = self.snake[0]
        apple_row, apple_col = self.apple
        
        # Direzione one-hot (4 elementi)
        direction_state = [
            1 if self.direction == (-1, 0) else 0,  # UP
            1 if self.direction == (1, 0) else 0,   # DOWN
            1 if self.direction == (0, -1) else 0,  # LEFT
            1 if self.direction == (0, 1) else 0    # RIGHT
        ]
        
        # Posizione mela relativa (2 elementi, normalizzati)
        apple_delta_row = (apple_row - head_row) / self.rows
        apple_delta_col = (apple_col - head_col) / self.cols
        
        # Pericoli intorno (4 elementi): up, down, left, right
        dangers = [
            1 if (head_row - 1, head_col) in self.snake or head_row - 1 < 0 else 0,  # UP
            1 if (head_row + 1, head_col) in self.snake or head_row + 1 >= self.rows else 0,  # DOWN
            1 if (head_row, head_col - 1) in self.snake or head_col - 1 < 0 else 0,  # LEFT
            1 if (head_row, head_col + 1) in self.snake or head_col + 1 >= self.cols else 0  # RIGHT
        ]
        
        # Lunghezza serpente normalizzata (1 elemento)
        # Max lunghezza possibile è rows*cols, ma normalizziamo a 0-1
        max_length = self.rows * self.cols
        snake_length = len(self.snake) / max_length
        
        state = np.array(
            direction_state + [apple_delta_row, apple_delta_col] + dangers + [snake_length],
            dtype=np.float32
        )
        
        return state
    
    def render(self):
        """Stampa una rappresentazione testuale del gioco (opzionale)"""
        grid = [['.' for _ in range(self.cols)] for _ in range(self.rows)]
        
        for segment in self.snake:
            grid[segment[0]][segment[1]] = 'S'
        
        grid[self.snake[0][0]][self.snake[0][1]] = 'H'  # Testa
        grid[self.apple[0]][self.apple[1]] = 'A'  # Mela
        
        for row in grid:
            print(''.join(row))
        print()
