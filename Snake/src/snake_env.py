import random
import numpy as np
from collections import deque


class SnakeAIEnv:
    """
    Ambiente Snake — stato CNN (3, ROWS, COLS).

    Canali:
      0 → corpo con gradiente: testa=1.0, ogni segmento scende verso 0
      1 → testa: 1.0 solo sulla cella testa
      2 → mela:  1.0 solo sulla cella mela

    Reward serpentina (aggiunto):
      Il problema diagnosticato dai log è che il serpente muore sempre
      per self_collision subito dopo aver mangiato, perché non pianifica
      dove finirà il corpo. La soluzione è incentivare il movimento
      a serpentina (righe orizzontali alternate), che è il pattern che
      minimizza il rischio di intrappolamento a corpo lungo.

      Come funziona la serpentina reward:
        1. STREAK BONUS (+0.15/step): ogni step consecutivo nella stessa
           direzione aumenta un contatore. Bonus piccolo per ogni step
           che mantiene la direzione. Incentiva le "righe" della serpentina.
           Cappato a COLS-1 step per non premiare loop infiniti.

        2. TURN BONUS (+0.4): quando il serpente gira di 90° dopo aver
           percorso almeno MIN_STRAIGHT=4 step in linea retta, riceve
           un bonus. Premia il completamento di una riga e la svolta
           ordinata — esattamente il pattern di una serpentina.

      Perché questi valori:
        - 0.15 e 0.4 sono piccoli rispetto a +20 (mela) e -10 (morte)
          → non sovrascrivono il comportamento base, solo lo sfumano
        - streak cappato a COLS-1 (16) → max bonus riga = 16 * 0.15 = 2.4
          che è inferiore al reward mela, quindi la mela resta priorità
        - MIN_STRAIGHT=4 → evita di premiare svolte rapide casuali
    """

    MIN_STRAIGHT = 4      # step minimi in linea retta per ottenere turn bonus
    STREAK_BONUS = 0.15   # bonus per ogni step nella stessa direzione
    TURN_BONUS   = 0.40   # bonus per svolta dopo riga lunga

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

        # — Stato serpentina ——————————————————————————————
        self.straight_streak   = 0   # step consecutivi nella stessa direzione
        self.last_direction    = (0, 1)

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

    def _serpentine_reward(self, new_direction):
        """
        Calcola il reward serpentina confrontando la nuova direzione
        con quella precedente.

        Casi:
          - Stessa direzione  → streak++, piccolo bonus
          - Svolta 90° dopo streak lungo → turn bonus, reset streak
          - Svolta 90° prematura → reset streak, nessun bonus
          - Inversione (180°) → reset streak, nessun bonus
            (le inversioni sono già bloccate dalla direction_map)
        """
        bonus = 0.0

        if new_direction == self.last_direction:
            # Stessa direzione: accumula streak
            self.straight_streak += 1
            # Cappato a COLS-1 per non premiare loop infiniti
            capped = min(self.straight_streak, self.cols - 1)
            bonus  = self.STREAK_BONUS * (capped / (self.cols - 1))
            # Nota: scala linearmente con la lunghezza della riga percorsa
            # → una riga completa vale STREAK_BONUS, mezza riga vale STREAK_BONUS/2

        else:
            # Direzione cambiata
            # Controlla se è una svolta 90° (non un'inversione)
            dr_old, dc_old = self.last_direction
            dr_new, dc_new = new_direction
            is_turn = (dr_old * dr_new + dc_old * dc_new == 0)  # prodotto scalare = 0

            if is_turn and self.straight_streak >= self.MIN_STRAIGHT:
                bonus = self.TURN_BONUS

            # Reset streak in ogni caso di cambio direzione
            self.straight_streak = 0

        self.last_direction = new_direction
        return bonus

    def step(self, action):
        direction_map = {
            (0,  1): [(-1, 0), (0,  1), ( 1, 0)],
            (0, -1): [( 1, 0), (0, -1), (-1, 0)],
            ( 1, 0): [(0,  1), ( 1, 0), (0, -1)],
            (-1, 0): [(0, -1), (-1, 0), (0,  1)],
        }

        new_direction  = direction_map[self.direction][action]
        self.direction = new_direction

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
            # Serpentina reward anche sulla mela
            reward += self._serpentine_reward(new_direction)

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

            # Serpentina reward sul movimento normale
            reward += self._serpentine_reward(new_direction)

        self.steps += 1
        if self.steps > 2000:
            done = True

        return self._get_state(), reward, done

    def _get_state(self):
        """
        Ritorna un array (3, rows, cols) float32.

        Canale 0 — corpo gradiente
        Canale 1 — testa
        Canale 2 — mela
        """
        state = np.zeros((3, self.rows, self.cols), dtype=np.float32)

        body_len = len(self.snake)
        for i, (r, c) in enumerate(self.snake):
            val = 1.0 - (i / max(body_len, 1)) * 0.9
            state[0, r, c] = val

        head_r, head_c = self.snake[0]
        state[1, head_r, head_c] = 1.0

        apple_r, apple_c = self.apple
        state[2, apple_r, apple_c] = 1.0

        return state