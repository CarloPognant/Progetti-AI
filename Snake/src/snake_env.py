import random
import numpy as np


class SnakeAIEnv:
    """
    Ambiente Snake con stato esteso (20 valori) e reward shaping migliorato.
    """

    def __init__(self, rows=15, cols=17):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        self.snake           = [(self.rows // 2, self.cols // 2)]
        self.direction       = (0, 1)
        self.apple           = self._spawn_apple()
        self.steps           = 0
        self.score           = 0
        self.steps_since_apple = 0
        return self._get_state()

    def _spawn_apple(self):
        while True:
            apple = (random.randint(0, self.rows - 1),
                     random.randint(0, self.cols - 1))
            if apple not in self.snake:
                return apple

    def step(self, action):
        """
        action: 0 = sinistra relativa, 1 = dritto, 2 = destra relativa
        """
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

        # — Collisione muro ——————————————————————————
        if not (0 <= new_head[0] < self.rows and 0 <= new_head[1] < self.cols):
            reward = -10.0
            done   = True

        # — Collisione se stesso ——————————————————————
        elif new_head in self.snake:
            reward = -10.0
            done   = True

        # — Mangia mela ——————————————————————————————
        elif new_head == self.apple:
            self.snake.insert(0, new_head)
            self.apple = self._spawn_apple()
            reward = 20.0
            self.score += 1
            self.steps_since_apple = 0

        # — Movimento normale ————————————————————————
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            self.steps_since_apple += 1

            # Reward di avvicinamento/allontanamento dalla mela
            new_dist = abs(new_head[0] - apple_row) + abs(new_head[1] - apple_col)
            if new_dist < old_dist:
                reward = 0.1    # si avvicina
            else:
                reward = -0.1   # si allontana

            # Penalità per girare in cerchio
            if self.steps_since_apple > self.rows * self.cols:
                reward = -5.0
                done   = True

        self.steps += 1
        if self.steps > 2000:
            done = True

        return self._get_state(), reward, done

    # ——————————————————————————————————————————————————————————
    def _get_state(self):
        """
        Stato esteso: 20 valori
          [0-3]   direzione one-hot (su, giù, sx, dx)
          [4-5]   delta mela normalizzato (riga, col)
          [6-9]   pericolo assoluto 4 direzioni
          [10-13] pericolo relativo (sx, dritto, dx, dietro)
          [14-17] distanza dai muri normalizzata (su, giù, sx, dx)
          [18]    lunghezza serpente normalizzata
          [19]    distanza manhattan dalla mela normalizzata
        """
        head_row, head_col = self.snake[0]
        apple_row, apple_col = self.apple
        dr, dc = self.direction

        # ── direzione one-hot ──────────────────────────────────
        direction_state = [
            1.0 if self.direction == (-1, 0) else 0.0,  # su
            1.0 if self.direction == ( 1, 0) else 0.0,  # giù
            1.0 if self.direction == (0, -1) else 0.0,  # sinistra
            1.0 if self.direction == (0,  1) else 0.0,  # destra
        ]

        # ── delta mela ────────────────────────────────────────
        apple_delta_row = float(apple_row - head_row) / self.rows
        apple_delta_col = float(apple_col - head_col) / self.cols

        # ── pericolo assoluto (4 celle adiacenti) ──────────────
        def is_dangerous(r, c):
            return (r < 0 or r >= self.rows or
                    c < 0 or c >= self.cols or
                    (r, c) in self.snake)

        dangers_abs = [
            1.0 if is_dangerous(head_row - 1, head_col) else 0.0,  # su
            1.0 if is_dangerous(head_row + 1, head_col) else 0.0,  # giù
            1.0 if is_dangerous(head_row, head_col - 1) else 0.0,  # sx
            1.0 if is_dangerous(head_row, head_col + 1) else 0.0,  # dx
        ]

        # ── pericolo relativo (rispetto alla direzione corrente) ─
        left_dir  = ( dc, -dr)   # ruota a sinistra
        right_dir = (-dc,  dr)   # ruota a destra
        back_dir  = (-dr, -dc)   # dietro

        dangers_rel = [
            1.0 if is_dangerous(head_row + left_dir[0],  head_col + left_dir[1])  else 0.0,
            1.0 if is_dangerous(head_row + dr,           head_col + dc)            else 0.0,
            1.0 if is_dangerous(head_row + right_dir[0], head_col + right_dir[1]) else 0.0,
            1.0 if is_dangerous(head_row + back_dir[0],  head_col + back_dir[1])  else 0.0,
        ]

        # ── distanza dai muri ──────────────────────────────────
        dist_up    = float(head_row)                  / self.rows
        dist_down  = float(self.rows - 1 - head_row)  / self.rows
        dist_left  = float(head_col)                  / self.cols
        dist_right = float(self.cols - 1 - head_col)  / self.cols

        # ── lunghezza serpente ─────────────────────────────────
        snake_length = float(len(self.snake)) / (self.rows * self.cols)

        # ── distanza manhattan dalla mela ──────────────────────
        manhattan = float(abs(apple_row - head_row) + abs(apple_col - head_col))
        manhattan /= (self.rows + self.cols)

        state = np.array(
            direction_state +
            [apple_delta_row, apple_delta_col] +
            dangers_abs +
            dangers_rel +
            [dist_up, dist_down, dist_left, dist_right] +
            [snake_length, manhattan],
            dtype=np.float32
        )

        assert len(state) == 20, f"Stato ha {len(state)} valori invece di 20!"
        return state