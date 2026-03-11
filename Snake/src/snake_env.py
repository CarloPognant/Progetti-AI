import random
import numpy as np
from collections import deque


class SnakeAIEnv:
    """
    Ambiente Snake con stato esteso (22 valori) + flood fill + loop detection.
    """

    def __init__(self, rows=15, cols=17):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        self.snake             = [(self.rows // 2, self.cols // 2)]
        self.direction         = (0, 1)
        self.apple             = self._spawn_apple()
        self.steps             = 0
        self.score             = 0
        self.steps_since_apple = 0
        self.visited_positions = {}   # posizione → quante volte visitata
        return self._get_state()

    def _spawn_apple(self):
        while True:
            apple = (random.randint(0, self.rows - 1),
                     random.randint(0, self.cols - 1))
            if apple not in self.snake:
                return apple

    def _flood_fill(self, start):
        """
        Conta quante celle sono raggiungibili da start
        senza passare attraverso il corpo del serpente.
        """
        visited   = set()
        queue     = deque([start])
        snake_set = set(self.snake)

        while queue:
            cell = queue.popleft()
            if cell in visited:
                continue
            r, c = cell
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                continue
            if cell in snake_set:
                continue
            visited.add(cell)
            queue.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])

        return len(visited)

    def step(self, action):
        direction_map = {
            (0,  1): [(-1, 0), (0,  1), ( 1, 0)],
            (0, -1): [( 1, 0), (0, -1), (-1, 0)],
            ( 1, 0): [(0,  1), ( 1, 0), (0, -1)],
            (-1, 0): [(0, -1), (-1, 0), (0,  1)],
        }

        self.direction = direction_map[self.direction][action]
        head_row, head_col = self.snake[0]
        new_head = (head_row + self.direction[0],
                    head_col + self.direction[1])

        apple_row, apple_col = self.apple
        old_dist = abs(head_row - apple_row) + abs(head_col - apple_col)

        reward = 0.0
        done   = False

        # — Collisione muro ————————————————————————————————
        if not (0 <= new_head[0] < self.rows and 0 <= new_head[1] < self.cols):
            reward = -10.0
            done   = True

        # — Collisione se stesso ———————————————————————————
        elif new_head in self.snake:
            reward = -10.0
            done   = True

        # — Mangia mela ————————————————————————————————————
        elif new_head == self.apple:
            self.snake.insert(0, new_head)
            self.apple = self._spawn_apple()
            reward = 20.0
            self.score += 1
            self.steps_since_apple = 0
            self.visited_positions = {}   # reset visite dopo ogni mela

        # — Movimento normale ——————————————————————————————
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            self.steps_since_apple += 1

            # 🎯 Reward avvicinamento/allontanamento (AUMENTATO 10x!)
            new_dist = abs(new_head[0] - apple_row) + abs(new_head[1] - apple_col)
            reward = 1.0 if new_dist < old_dist else -1.0

            # ⚠️ Penalità loop RIDOTTA (solo dalla 3a visita)
            visit_count = self.visited_positions.get(new_head, 0) + 1
            self.visited_positions[new_head] = visit_count
            if visit_count >= 3:
                reward -= 0.5 * (visit_count - 2)

            # ⚠️ Penalità trappola RIDOTTA (da 3.0 a 1.0)
            free_space  = self._flood_fill(new_head)
            body_length = len(self.snake)
            if free_space < body_length:
                trap_ratio = free_space / max(body_length, 1)
                reward -= (1.0 - trap_ratio) * 1.0

            # Timeout RIDOTTO (da 127 a 100 step)
            if self.steps_since_apple > 100:
                reward = -5.0
                done   = True

        self.steps += 1
        if self.steps > 2000:
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        """
        Stato: 22 valori
          [0-3]   direzione one-hot
          [4-5]   delta mela normalizzato
          [6-9]   pericolo assoluto 4 direzioni
          [10-13] pericolo relativo (sx, dritto, dx, dietro)
          [14-17] distanza dai muri normalizzata
          [18]    lunghezza serpente normalizzata
          [19]    distanza manhattan dalla mela normalizzata
          [20]    spazio libero normalizzato (flood fill)
          [21]    rapporto spazio/corpo (< 1 = trappola)
        """
        head_row, head_col = self.snake[0]
        apple_row, apple_col = self.apple
        dr, dc = self.direction

        direction_state = [
            1.0 if self.direction == (-1, 0) else 0.0,
            1.0 if self.direction == ( 1, 0) else 0.0,
            1.0 if self.direction == (0, -1) else 0.0,
            1.0 if self.direction == (0,  1) else 0.0,
        ]

        apple_delta_row = float(apple_row - head_row) / self.rows
        apple_delta_col = float(apple_col - head_col) / self.cols

        def is_dangerous(r, c):
            return (r < 0 or r >= self.rows or
                    c < 0 or c >= self.cols or
                    (r, c) in self.snake)

        dangers_abs = [
            1.0 if is_dangerous(head_row - 1, head_col) else 0.0,
            1.0 if is_dangerous(head_row + 1, head_col) else 0.0,
            1.0 if is_dangerous(head_row, head_col - 1) else 0.0,
            1.0 if is_dangerous(head_row, head_col + 1) else 0.0,
        ]

        left_dir  = ( dc, -dr)
        right_dir = (-dc,  dr)
        back_dir  = (-dr, -dc)

        dangers_rel = [
            1.0 if is_dangerous(head_row + left_dir[0],  head_col + left_dir[1])  else 0.0,
            1.0 if is_dangerous(head_row + dr,           head_col + dc)            else 0.0,
            1.0 if is_dangerous(head_row + right_dir[0], head_col + right_dir[1]) else 0.0,
            1.0 if is_dangerous(head_row + back_dir[0],  head_col + back_dir[1])  else 0.0,
        ]

        dist_up    = float(head_row)                 / self.rows
        dist_down  = float(self.rows - 1 - head_row) / self.rows
        dist_left  = float(head_col)                 / self.cols
        dist_right = float(self.cols - 1 - head_col) / self.cols

        total_cells  = self.rows * self.cols
        snake_length = float(len(self.snake)) / total_cells

        manhattan  = float(abs(apple_row - head_row) + abs(apple_col - head_col))
        manhattan /= (self.rows + self.cols)

        free_space = self._flood_fill(self.snake[0])
        free_norm  = float(free_space) / total_cells
        trap_ratio = float(free_space) / max(len(self.snake), 1)
        trap_ratio = min(trap_ratio, 5.0) / 5.0

        state = np.array(
            direction_state +
            [apple_delta_row, apple_delta_col] +
            dangers_abs +
            dangers_rel +
            [dist_up, dist_down, dist_left, dist_right] +
            [snake_length, manhattan] +
            [free_norm, trap_ratio],
            dtype=np.float32
        )

        assert len(state) == 22, f"Stato ha {len(state)} valori invece di 22!"
        return state