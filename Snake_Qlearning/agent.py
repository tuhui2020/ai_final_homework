from collections import defaultdict
import numpy as np

class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.action_size = 3  # 前进 / 左转 / 右转
        self.Q = defaultdict(lambda: np.zeros(self.action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action]
        )
        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay