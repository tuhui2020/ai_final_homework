import os
import pickle
from agent import QLearningAgent
from env import SnakeEnv

# 保存 / 加载 Q 表
def save_q(agent, path):
    with open(path, 'wb') as f:
        pickle.dump(dict(agent.Q), f)

def load_q(agent, path):
    with open(path, 'rb') as f:
        q = pickle.load(f)
    agent.Q.update(q)

# 训练流程
def train():
    env = SnakeEnv(grid_size=10)

    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, 'best_q.pkl')

    episodes = 2000  # 可以增加训练轮数
    best_score = -float('inf')

    for ep in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            score += reward

        if score > best_score:
            best_score = score
            save_q(agent, best_path)

        print(f"Episode {ep + 1}/{episodes}, score = {score}, epsilon = {agent.epsilon:.3f}")

    print("Training finished. Best score:", best_score)

if __name__ == '__main__':
    train()