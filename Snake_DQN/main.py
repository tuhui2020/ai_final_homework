import torch
from env import SnakeEnv
from dqn_agent import DQNAgent
from config import NUM_EPISODES, TARGET_UPDATE

env = SnakeEnv()
state_dim = len(env.get_state())
action_dim = 4
agent = DQNAgent(state_dim, action_dim)

reward_history = []

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward

    if episode % TARGET_UPDATE == 0:
        agent.update_target()

    reward_history.append(total_reward)

    # 每 50 局打印一次平均分和当前 epsilon
    if episode % 50 == 0:
        avg_score = sum(reward_history[-50:]) / 50
        print(f"Episode {episode}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

    # 每 500 局保存一次模型
    if episode % 500 == 0:
        torch.save(agent.policy_net.state_dict(), f"snake_dqn_ep{episode}.pth")
        print(f"Saved model at Episode {episode}")

# 最终训练结束后保存
torch.save(agent.policy_net.state_dict(), "snake_dqn_final.pth")
print("Training finished, final model saved.")
