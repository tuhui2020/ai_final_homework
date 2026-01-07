# env.py
import random
from config import GRID_SIZE, REWARD_FOOD, REWARD_DEAD, REWARD_STEP, MAX_STEPS

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

class SnakeEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]
        self.direction = RIGHT
        self._place_food()
        self.done = False
        self.steps = 0
        return self.get_state()

    def _place_food(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1),
                   random.randint(0, self.grid_size - 1))
            if pos not in self.snake:
                self.food = pos
                break

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        # 禁止反向
        if (self.direction == UP and action == DOWN) or \
           (self.direction == DOWN and action == UP) or \
           (self.direction == LEFT and action == RIGHT) or \
           (self.direction == RIGHT and action == LEFT):
            action = self.direction

        self.direction = action
        head_x, head_y = self.snake[0]

        # 移动蛇头
        if action == UP:
            new_head = (head_x - 1, head_y)
        elif action == DOWN:
            new_head = (head_x + 1, head_y)
        elif action == LEFT:
            new_head = (head_x, head_y - 1)
        else:
            new_head = (head_x, head_y + 1)

        self.steps += 1
        reward = REWARD_STEP

        # 撞墙或撞自己
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            self.done = True
            return self.get_state(), REWARD_DEAD, True

        self.snake.insert(0, new_head)

        # 吃食物
        if new_head == self.food:
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()

        if self.steps >= MAX_STEPS:
            self.done = True

        return self.get_state(), reward, self.done

    def get_state(self):
        """增强状态表示：
        [蛇头x, 蛇头y, 食物dx, 食物dy, danger_up, danger_down, danger_left, danger_right]
        danger_* = 1 if撞墙或撞自己 else 0
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        food_dx = food_x - head_x
        food_dy = food_y - head_y

        # 检测危险
        danger_up = 1 if (head_x - 1 < 0 or (head_x - 1, head_y) in self.snake) else 0
        danger_down = 1 if (head_x + 1 >= self.grid_size or (head_x + 1, head_y) in self.snake) else 0
        danger_left = 1 if (head_y - 1 < 0 or (head_x, head_y - 1) in self.snake) else 0
        danger_right = 1 if (head_y + 1 >= self.grid_size or (head_x, head_y + 1) in self.snake) else 0

        state = [head_x, head_y, food_dx, food_dy, danger_up, danger_down, danger_left, danger_right]
        return state
