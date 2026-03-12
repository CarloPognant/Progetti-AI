"""
SumTree - Struttura dati ottimizzata per Prioritized Experience Replay

Complessità:
- add():    O(log N)
- sample(): O(log N) 
- update(): O(log N)

vs implementazione naive O(N) che causa il crollo di velocità
"""

import numpy as np


class SumTree:
    """
    Binary tree dove ogni nodo contiene la somma delle priorità dei figli.
    Le foglie contengono le priorità delle esperienze.
    
    Struttura:
                  sum=42
                 /      \
            sum=19      sum=23
           /    \       /    \
          p=7   p=12  p=15   p=8
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Nodi dell'albero
        self.data = np.zeros(capacity, dtype=object)  # Esperienze
        self.write = 0  # Puntatore circolare
        self.n_entries = 0  # Numero di elementi inseriti
        
    def _propagate(self, idx, change):
        """Propaga il cambio di priorità ai nodi genitori"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """
        Trova la foglia con priorità cumulativa = s
        
        Args:
            idx: Nodo corrente
            s: Valore target cumulativo
            
        Returns:
            Indice della foglia trovata
        """
        left = 2 * idx + 1
        right = left + 1
        
        # Se siamo a una foglia
        if left >= len(self.tree):
            return idx
        
        # Scendi a sinistra se la somma sinistra >= s
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            # Altrimenti scendi a destra (sottraendo la somma sinistra)
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Ritorna la somma totale delle priorità"""
        return self.tree[0]
    
    def add(self, priority, data):
        """
        Aggiunge esperienza con priorità
        
        Args:
            priority: Priorità dell'esperienza
            data: Tupla (state, action, reward, next_state, done)
        """
        idx = self.write + self.capacity - 1  # Converti a indice foglia
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """
        Aggiorna priorità di una foglia
        
        Args:
            idx: Indice foglia nell'albero
            priority: Nuova priorità
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """
        Ottieni esperienza con priorità cumulativa = s
        
        Args:
            s: Valore cumulativo target
            
        Returns:
            (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    Replay Buffer con Prioritized Experience Replay usando SumTree
    
    100x più veloce della implementazione naive quando piena!
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: Dimensione massima buffer
            alpha: Quanto usare le priorità (0=uniform, 1=full priority)
            beta_start: Valore iniziale beta per importance sampling
            beta_frames: Numero di frame per portare beta a 1.0
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # Piccolo valore per evitare priorità zero
    
    def _get_priority(self, error):
        """Converte errore TD in priorità"""
        return (np.abs(error) + self.epsilon) ** self.alpha
    
    def push(self, experience):
        """
        Aggiunge esperienza al buffer
        
        Args:
            experience: Tupla (state, action, reward, next_state, done)
        """
        # Nuove esperienze ottengono priorità massima
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        
        self.tree.add(max_priority, experience)
    
    def sample(self, batch_size):
        """
        Campiona batch con priorità
        
        Args:
            batch_size: Numero di esperienze da campionare
            
        Returns:
            batch: Lista di esperienze
            indices: Indici nell'albero
            weights: Importance sampling weights
        """
        batch = []
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float32)
        
        # Dividi [0, total_priority) in batch_size segmenti
        segment = self.tree.total() / batch_size
        
        # Calcola beta corrente (cresce linearmente a 1.0)
        beta = min(1.0, self.beta_start + 
                   (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        
        # Campiona un'esperienza da ogni segmento
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            (idx, priority, data) = self.tree.get(s)
            
            batch.append(data)
            indices[i] = idx
            priorities[i] = priority
        
        # Calcola importance sampling weights
        sampling_probabilities = priorities / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        weights /= weights.max()  # Normalizza
        
        return batch, indices, weights
    
    def update_priorities(self, indices, errors):
        """
        Aggiorna priorità delle esperienze
        
        Args:
            indices: Indici nell'albero
            errors: Errori TD
        """
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
    
    def __len__(self):
        """Ritorna numero di esperienze nel buffer"""
        return self.tree.n_entries