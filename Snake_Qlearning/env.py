import random

class SnakeEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (1, 0)  # 初始向右
        self.food = self._spawn_food()
        self.done = False
        return self.get_state()

    def _spawn_food(self):
        while True:
            f = (random.randint(0, self.grid_size - 1), 
                 random.randint(0, self.grid_size - 1))
            if f not in self.snake:
                return f

    def get_state(self):
        # 状态: (蛇头x, 蛇头y, 食物dx, 食物dy, 前方/左/右障碍)
        head_x, head_y = self.snake[0]
        food_dx = self.food[0] - head_x
        food_dy = self.food[1] - head_y

        # 检测前方/左/右是否有障碍
        def check(pos):
            x, y = pos
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                return 1
            if (x, y) in self.snake:
                return 1
            return 0

        dx, dy = self.direction
        front = check((head_x + dx, head_y + dy))
        left = check((head_x - dy, head_y + dx))
        right = check((head_x + dy, head_y - dx))

        return (head_x, head_y, food_dx, food_dy, front, left, right)

    def step(self, action):
        # 0=前进, 1=左转, 2=右转
        self._update_direction(action)
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        reward = -0.1  # 每走一步小惩罚
        self.done = False

        if new_head == self.food:
            reward = 10
            self.snake.insert(0, new_head)
            self.food = self._spawn_food()
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()

        # 撞墙或撞自己
        if not (0 <= new_head[0] < self.grid_size and 
                0 <= new_head[1] < self.grid_size) or \
                new_head in self.snake[1:]:
            reward = -10
            self.done = True

        return self.get_state(), reward, self.done

    def _update_direction(self, action):
        dx, dy = self.direction
        if action == 1:  # 左转
            self.direction = (-dy, dx)
        elif action == 2:  # 右转
            self.direction = (dy, -dx)
        # action == 0: 保持当前方向