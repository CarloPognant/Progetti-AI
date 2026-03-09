import random
import numpy as np

class SnakeAIEnv:
    """
    Ambiente MIGLIORATO per allenare l'AI del Snake
    """
    
    def __init__(self, rows=15, cols=17):
        self.rows = rows
        self.cols = cols
        self.reset()
    
    def reset(self):
        """Resetta il gioco"""
        self.snake = [(self.rows // 2, self.cols // 2)]
        self.direction = (0, 1)
        self.apple = self._spawn_apple()
        self.steps = 0
        self.score = 0
        self.steps_since_apple = 0
        return self._get_state()
    
    def _spawn_apple(self):
        """Genera una nuova mela"""
        while True:
            apple = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
            if apple not in self.snake:
                return apple
    
    def step(self, action):
        """
        Esegue un passo del gioco
        action: 0 = sinistra, 1 = dritto, 2 = destra
        """
        direction_map = {
            (0, 1): [(-1, 0), (0, 1), (1, 0)],
            (0, -1): [(1, 0), (0, -1), (-1, 0)],
            (1, 0): [(0, 1), (1, 0), (0, -1)],
            (-1, 0): [(0, -1), (-1, 0), (0, 1)]
        }
        
        self.direction = direction_map[self.direction][action]
        head_row, head_col = self.snake[0]
        new_head = (head_row + self.direction[0], head_col + self.direction[1])
        
        reward = 0.05
        done = False
        
        # Collisione muro
        if not (0 <= new_head[0] < self.rows) or not (0 <= new_head[1] < self.cols):
            reward = -1.0
            done = True
        # Collisione se stesso
        elif new_head in self.snake:
            reward = -1.0
            done = True
        # Mangia mela
        elif new_head == self.apple:
            self.snake.insert(0, new_head)
            self.apple = self._spawn_apple()
            reward = 50.0
            self.score += 1
            self.steps_since_apple = 0
        else:
            # Movimento normale
            self.snake.insert(0, new_head)
            self.snake.pop()
            self.steps_since_apple += 1
            
            if self.steps_since_apple > 100:
                reward = -0.1
        
        self.steps += 1
        if self.steps > 1000:
            done = True
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        """
        Ritorna lo stato come array numpy (11 elementi)
        """
        head_row, head_col = self.snake[0]
        apple_row, apple_col = self.apple
        
        direction_state = [
            1.0 if self.direction == (-1, 0) else 0.0,
            1.0 if self.direction == (1, 0) else 0.0,
            1.0 if self.direction == (0, -1) else 0.0,
            1.0 if self.direction == (0, 1) else 0.0
        ]
        
        apple_delta_row = float(apple_row - head_row) / self.rows
        apple_delta_col = float(apple_col - head_col) / self.cols
        
        dangers = [
            1.0 if (head_row - 1, head_col) in self.snake or head_row - 1 < 0 else 0.0,
            1.0 if (head_row + 1, head_col) in self.snake or head_row + 1 >= self.rows else 0.0,
            1.0 if (head_row, head_col - 1) in self.snake or head_col - 1 < 0 else 0.0,
            1.0 if (head_row, head_col + 1) in self.snake or head_col + 1 >= self.cols else 0.0
        ]
        
        max_length = self.rows * self.cols
        snake_length = float(len(self.snake)) / max_length
        
        state = np.array(
            direction_state + [apple_delta_row, apple_delta_col] + dangers + [snake_length],
            dtype=np.float32
        )
        
        return state