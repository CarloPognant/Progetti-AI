import random
import numpy as np
from collections import deque


class SnakeAIEnv:
    """
    Ambiente Snake — stato CNN (3, ROWS, COLS).

    Canali:
      0 → corpo con gradiente: testa=1.0, ogni segmento scende verso 0
          → la rete capisce forma e direzione di movimento del corpo
      1 → testa: 1.0 solo sulla cella testa
      2 → mela:  1.0 solo sulla cella mela

    Fix reward inclusi:
      - Trap penalty scalata con lunghezza corpo (1x → 4x)
      - Bug manhattan fix (era apple_col - apple_col, sempre 0)
      - Rimosso length_bonus che incentivava greedy behaviour
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
        self.visited_positions = {}
        return self._get_state()

    def _spawn_apple(self):
        while True:
            apple = (random.randint(0, self.rows - 1),
                     random.randint(0, self.cols - 1))
            if apple not in self.snake:
                return apple

    def _flood_fill(self, start):
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
            self.visited_positions = {}

        # — Movimento normale ——————————————————————————————
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            self.steps_since_apple += 1

            new_dist = abs(new_head[0] - apple_row) + abs(new_head[1] - apple_col)
            reward = 1.0 if new_dist < old_dist else -1.0

            # Penalità loop
            visit_count = self.visited_positions.get(new_head, 0) + 1
            self.visited_positions[new_head] = visit_count
            if visit_count >= 3:
                reward -= 0.5 * (visit_count - 2)

            # Penalità trappola scalata con lunghezza corpo
            free_space  = self._flood_fill(new_head)
            body_length = len(self.snake)
            if free_space < body_length * 1.5:
                trap_ratio    = free_space / max(body_length, 1)
                severity      = max(0.0, 1.0 - trap_ratio / 1.5)
                length_factor = min(1.0 + (body_length - 5) / 12.0, 4.0)
                length_factor = max(length_factor, 1.0)
                reward -= severity * length_factor

            if self.steps_since_apple > 100:
                reward = -5.0
                done   = True

        self.steps += 1
        if self.steps > 2000:
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        """
        Ritorna un array (3, rows, cols) float32.

        Canale 0 — corpo gradiente:
          Ogni segmento del corpo riceve un valore da 1.0 (testa) a ~0.1 (coda).
          Questo permette alla CNN di capire non solo dove è il corpo,
          ma anche da che parte si sta muovendo (il gradiente indica la direzione).

        Canale 1 — testa:
          Solo la cella della testa vale 1.0. Serve come ancora esplicita per
          orientare la rete su "dove sono io adesso".

        Canale 2 — mela:
          Solo la cella della mela vale 1.0. Obiettivo esplicito e spaziale.
        """
        state = np.zeros((3, self.rows, self.cols), dtype=np.float32)

        # Canale 0: gradiente corpo (testa=1.0, coda→min_val)
        body_len = len(self.snake)
        for i, (r, c) in enumerate(self.snake):
            # 1.0 alla testa, decade linearmente verso la coda
            val = 1.0 - (i / max(body_len, 1)) * 0.9   # range [1.0, 0.1]
            state[0, r, c] = val

        # Canale 1: testa
        head_r, head_c = self.snake[0]
        state[1, head_r, head_c] = 1.0

        # Canale 2: mela
        apple_r, apple_c = self.apple
        state[2, apple_r, apple_c] = 1.0

        return state