import torch
import random
from model import SnakeNet
from snake_env import snakeAIEnv

env = snakeAIEnv()
model = SnakeNet()

for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        action = random.randint(0, 2)  # Random action for exploration
        next_state, reward, done = env.step(action)
        state = next_state
    print("episode finito")