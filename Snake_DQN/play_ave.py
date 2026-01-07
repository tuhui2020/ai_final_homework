# play.py
import torch
from env import SnakeEnv
from dqn_agent import DQNAgent

NUM_EPISODES = 100  # 模拟多少局
PRINT_INTERVAL = 10  # 每多少局打印一次平均值

# 初始化环境
env = SnakeEnv()
state_dim = len(env.get_state())
action_dim = 4

# 初始化agent
agent = DQNAgent(state_dim, action_dim)

# 加载训练好的模型
agent.policy_net.load_state_dict(torch.load("snake_dqn_ep1500.pth"))
agent.policy_net.eval()

def select_action_greedy(agent, state):
    """完全贪心策略，不探索"""
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return agent.policy_net(state).argmax(dim=1).item()

total_rewards = []
total_foods = []

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    done = False
    episode_reward = 0
    food_count = 0

    while not done:
        action = select_action_greedy(agent, state)
        next_state, reward, done = env.step(action)
        state = next_state
        episode_reward += reward
        if reward > 0:  # 吃到食物
            food_count += 1

    total_rewards.append(episode_reward)
    total_foods.append(food_count)

    # 每 PRINT_INTERVAL 局打印一次平均值
    if episode % PRINT_INTERVAL == 0:
        avg_reward = sum(total_rewards[-PRINT_INTERVAL:]) / PRINT_INTERVAL
        avg_food = sum(total_foods[-PRINT_INTERVAL:]) / PRINT_INTERVAL
        print(f"Episodes {episode-PRINT_INTERVAL+1}-{episode}: "
              f"Avg Reward = {avg_reward:.2f}, Avg Food = {avg_food:.2f}")

# 最终统计
print(f"\nSimulated {NUM_EPISODES} episodes")
print(f"Average total reward: {sum(total_rewards)/NUM_EPISODES:.2f}")
print(f"Average food eaten: {sum(total_foods)/NUM_EPISODES:.2f}")
print(f"Max food eaten: {max(total_foods)}")
print(f"Min food eaten: {min(total_foods)}")
