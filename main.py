import random
from enum import Enum

import numpy as np
import pyray as rl
from pyray import KeyboardKey

WIDTH = 800
HEIGHT = 600
SNAKE_SIZE = 20
SNAKE_SPEED = 20
CELLS = 20
CELL_SIZE = int(HEIGHT / CELLS)

rl.init_window(WIDTH, HEIGHT, "Snake Game")
rl.set_target_fps(60)


class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Snake:
    length = 2
    body = [(8, 8)]
    dir = Direction.RIGHT

    def resolve_direction(self, new_dir: Direction):
        dir = self.dir
        if new_dir.value[0] != 0 and dir.value[1] != 0:
            dir = new_dir
        elif new_dir.value[1] != 0 and dir.value[0] != 0:
            dir = new_dir
        self.dir = dir


grid = np.zeros((CELLS, CELLS), dtype=np.int8)
grid[3, 3] = -1

snake = Snake()

last = 0
steps = 0
new_dir = Direction.RIGHT
while not rl.window_should_close():
    match rl.get_key_pressed():
        case KeyboardKey.KEY_K:
            new_dir = Direction.UP
        case KeyboardKey.KEY_J:
            new_dir = Direction.DOWN
        case KeyboardKey.KEY_H:
            new_dir = Direction.LEFT
        case KeyboardKey.KEY_L:
            new_dir = Direction.RIGHT

    dt = rl.get_frame_time()
    last += dt
    if last > 0.100:
        step = True
        last = 0
    else:
        step = False

    if step:
        steps += 1
        snake.resolve_direction(new_dir)
        new_head = (snake.body[0][0] + snake.dir.value[0], snake.body[0][1] + snake.dir.value[1])

        if grid[new_head] == -1:
            snake.length += 1
            x = random.randint(0, CELLS - 1)
            y = random.randint(0, CELLS - 1)
            grid[x, y] = -1

        snake.body.insert(0, new_head)
        if len(snake.body) > snake.length:
            tail = snake.body.pop()
            grid[tail] = 0

        if new_head[0] < 0 or new_head[0] >= CELLS or new_head[1] < 0 or new_head[1] >= CELLS:
            break

        for i, segment in enumerate(snake.body):
            grid[segment] = snake.length - i

    rl.begin_drawing()
    rl.clear_background(rl.color_from_hsv(0, 0, 0.1))
    rl.draw_grid(CELLS, CELL_SIZE)
    for i, rows in enumerate(grid):
        for j, cell in enumerate(rows):
            if cell == -1:
                color = rl.GREEN
            elif cell >= 1:
                color = rl.color_from_hsv(55, 1, cell / snake.length)
            else:
                color = rl.color_from_hsv(0, 1, cell / snake.length)

            rl.draw_rectangle(
                j * CELL_SIZE,
                i * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
                color,
            )

    rl.draw_text(f"{steps} steps", HEIGHT + 8, 8, 24, rl.RAYWHITE)
    rl.draw_text(f"{snake.length} size", HEIGHT + 8, 32, 24, rl.RAYWHITE)
    rl.end_drawing()

rl.close_window()
