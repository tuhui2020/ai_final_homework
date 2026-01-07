# utils.py

UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


def turn_left(direction):
    idx = DIRECTIONS.index(direction)
    return DIRECTIONS[(idx - 1) % 4]


def turn_right(direction):
    idx = DIRECTIONS.index(direction)
    return DIRECTIONS[(idx + 1) % 4]


def move(pos, direction):
    x, y = pos
    dx, dy = direction
    return (x + dx, y + dy)


def danger_at(head, direction, snake, grid_size):
    nx, ny = move(head, direction)
    return (
        nx < 0 or nx >= grid_size or
        ny < 0 or ny >= grid_size or
        (nx, ny) in snake
    )


def food_direction(head, food):
    hx, hy = head
    fx, fy = food
    return (
        int(fx < hx),
        int(fx > hx),
        int(fy < hy),
        int(fy > hy)
    )


def encode_state(head, direction, food, snake, grid_size):
    straight = direction
    left = turn_left(direction)
    right = turn_right(direction)

    return (
        int(danger_at(head, straight, snake, grid_size)),
        int(danger_at(head, left, snake, grid_size)),
        int(danger_at(head, right, snake, grid_size)),
        *food_direction(head, food),
        direction
    )
