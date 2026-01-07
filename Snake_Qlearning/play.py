# play.py

import pygame
import pickle
import time

from env import SnakeEnv
from agent import QLearningAgent
from config import *

# =====================
# UI 参数
# =====================
CELL_SIZE = 30
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 10

# =====================
# 初始化 pygame
# =====================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Snake (Play)")
clock = pygame.time.Clock()

# =====================
# 加载训练好的 Q-table
# =====================
agent = QLearningAgent(ALPHA, GAMMA)

with open(f"{CHECKPOINT_DIR}/best_q_table.pkl", "rb") as f:
    agent.Q = pickle.load(f)

print("Loaded trained Q-table")

# =====================
# 初始化环境
# =====================
env = SnakeEnv(GRID_SIZE)
state = env.reset()

# =====================
# 主循环（只演示）
# =====================
running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 贪心策略（不探索）
    action = agent.choose_action(state, epsilon=0.0)
    state, _, done = env.step(action)

    if done:
        time.sleep(0.5)
        state = env.reset()

    # =====================
    # 画面渲染
    # =====================
    screen.fill((0, 0, 0))

    # 画食物（红色）
    fx, fy = env.food
    pygame.draw.rect(
        screen,
        (255, 0, 0),
        (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # 画蛇（绿色）
    for i, (x, y) in enumerate(env.snake):
        color = (0, 255, 0) if i == 0 else (0, 180, 0)
        pygame.draw.rect(
            screen,
            color,
            (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

    pygame.display.flip()

pygame.quit()
