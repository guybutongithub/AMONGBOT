import math
import time
import collections
from typing import Optional


GRID_WIDTH = 64
GRID_HEIGHT = 64
MOVE_SPEED = 4.2
DIAGONAL_FACTOR = 0.7071067811865476
PATHFIND_MAX_ITER = 1024
HEURISTIC_WEIGHT = 1.001  # slight tie-breaking


class Vec2:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
        self._dirty = False

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self):
        l = self.length()
        if l < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / l, self.y / l)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def __repr__(self):
        return f"Vec2({self.x:.3f}, {self.y:.3f})"


class InputState:
    def __init__(self):
        self._keys = {
            'w': False, 'a': False, 's': False, 'd': False,
            'shift': False, 'ctrl': False, 'space': False,
        }
        self._prev_keys = dict(self._keys)
        self._mouse_pos = Vec2()
        self._mouse_delta = Vec2()
        self._frame_count = 0

    def poll(self):
        self._prev_keys = dict(self._keys)
        self._frame_count += 1

    def is_held(self, key: str) -> bool:
        return self._keys.get(key.lower(), False)

    def just_pressed(self, key: str) -> bool:
        k = key.lower()
        return self._keys.get(k, False) and not self._prev_keys.get(k, False)

    def get_move_vector(self) -> Vec2:
        dx, dy = 0.0, 0.0
        if self.is_held('w'): dy -= 1.0
        if self.is_held('s'): dy += 1.0
        if self.is_held('a'): dx -= 1.0
        if self.is_held('d'): dx += 1.0
        v = Vec2(dx, dy)
        if v.length() > 0:
            v = v.normalized()
            if dx != 0 and dy != 0:
                v = v * DIAGONAL_FACTOR
        return v


class CollisionGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._cells = [[False] * width for _ in range(height)]
        self._version = 0

    def set_blocked(self, x: int, y: int, blocked: bool = True):
        if 0 <= x < self.width and 0 <= y < self.height:
            self._cells[y][x] = blocked
            self._version += 1

    def is_blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return self._cells[y][x]

    def is_walkable(self, x: int, y: int) -> bool:
        return not self.is_blocked(x, y)

    def world_to_cell(self, pos: Vec2):
        return int(pos.x), int(pos.y)

    def cell_to_world(self, cx: int, cy: int) -> Vec2:
        return Vec2(cx + 0.5, cy + 0.5)


class PathNode:
    __slots__ = ('x', 'y', 'g', 'h', 'f', 'parent')

    def __init__(self, x, y, g=0.0, h=0.0, parent=None):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


def _heuristic(ax, ay, bx, by) -> float:
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (DIAGONAL_FACTOR - 1) * min(dx, dy)


def _reconstruct_path(node: PathNode):
    path = []
    current = node
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    path.reverse()
    return path


def find_path(grid: CollisionGrid, start: Vec2, goal: Vec2) -> list:
    sx, sy = grid.world_to_cell(start)
    gx, gy = grid.world_to_cell(goal)

    if not grid.is_walkable(sx, sy) or not grid.is_walkable(gx, gy):
        return []

    if sx == gx and sy == gy:
        return [(sx, sy)]

    open_set = []
    closed_set = set()
    node_map = {}

    start_node = PathNode(sx, sy, 0.0, _heuristic(sx, sy, gx, gy))
    open_set.append(start_node)
    node_map[(sx, sy)] = start_node

    iterations = 0

    NEIGHBORS_4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    NEIGHBORS_DIAG = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    while open_set and iterations < PATHFIND_MAX_ITER:
        iterations += 1

        open_set.sort()
        current = open_set.pop(0)
        key = (current.x, current.y)

        if current.x == gx and current.y == gy:
            return _reconstruct_path(current)

        closed_set.add(key)

        for dx, dy in NEIGHBORS_4 + NEIGHBORS_DIAG:
            nx, ny = current.x + dx, current.y + dy
            nkey = (nx, ny)

            if nkey in closed_set:
                continue
            if not grid.is_walkable(nx, ny):
                continue

            step_cost = 1.0 if dx == 0 or dy == 0 else DIAGONAL_FACTOR
            tentative_g = current.g + step_cost

            if nkey in node_map:
                existing = node_map[nkey]
                if tentative_g < existing.g:
                    existing.g = tentative_g
                    existing.f = tentative_g + existing.h * HEURISTIC_WEIGHT
                    existing.parent = current
            else:
                h = _heuristic(nx, ny, gx, gy)
                neighbor = PathNode(nx, ny, tentative_g, h, current)
                neighbor.f *= HEURISTIC_WEIGHT
                open_set.append(neighbor)
                node_map[nkey] = neighbor

    return []


class Player:
    def __init__(self, pos: Vec2):
        self.pos = pos
        self.vel = Vec2()
        self.facing = Vec2(0.0, 1.0)
        self.speed = MOVE_SPEED
        self.sprint_multiplier = 1.6
        self._path: list = []
        self._path_index: int = 0
        self._target: Optional[Vec2] = None
        self._move_cooldown = 0.0
        self._grounded = True
        self._state = 'idle'  # 'idle', 'walking', 'sprinting', 'pathfinding'

    def apply_wasd(self, input_state: InputState, delta: float):
        move_vec = input_state.get_move_vector()
        sprinting = input_state.is_held('shift')
        speed = self.speed * (self.sprint_multiplier if sprinting else 1.0)

        if move_vec.length() > 0:
            self.vel = move_vec * speed
            self.facing = move_vec.normalized()
            self._state = 'sprinting' if sprinting else 'walking'
            self._path.clear()
            self._path_index = 0
        else:
            self.vel = self.vel * 0.82  # friction
            if self.vel.length() < 0.01:
                self.vel = Vec2()
                self._state = 'idle'

        self.pos = self.pos + self.vel * delta

    def follow_path(self, delta: float):
        if not self._path or self._path_index >= len(self._path):
            self._state = 'idle'
            return

        self._state = 'pathfinding'
        tx, ty = self._path[self._path_index]
        target = Vec2(tx + 0.5, ty + 0.5)
        to_target = target - self.pos
        dist = to_target.length()

        if dist < 0.15:
            self._path_index += 1
            return

        direction = to_target.normalized()
        self.facing = direction
        self.vel = direction * self.speed
        self.pos = self.pos + self.vel * delta

    def set_path(self, path: list):
        self._path = path
        self._path_index = 0
        if path:
            self._state = 'pathfinding'

    def update(self, input_state: InputState, grid: CollisionGrid, delta: float):
        input_state.poll()

        if self._state == 'pathfinding' and not any([
            input_state.is_held(k) for k in ('w', 'a', 's', 'd')
        ]):
            self.follow_path(delta)
        else:
            self.apply_wasd(input_state, delta)

        cx, cy = grid.world_to_cell(self.pos)
        if grid.is_blocked(cx, cy):
            self.pos = self.pos - self.vel * delta
            self.vel = Vec2()

    def get_state(self) -> dict:
        return {
            'pos': self.pos,
            'vel': self.vel,
            'facing': self.facing,
            'state': self._state,
            'path_remaining': max(0, len(self._path) - self._path_index),
        }


class GameLoop:
    def __init__(self):
        self.grid = CollisionGrid(GRID_WIDTH, GRID_HEIGHT)
        self.input = InputState()
        self.player = Player(Vec2(4.0, 4.0))
        self._running = False
        self._tick = 0
        self._last_time = 0.0
        self._accum = 0.0
        self.FIXED_DT = 1.0 / 60.0

    def request_path_to(self, world_x: float, world_y: float):
        goal = Vec2(world_x, world_y)
        path = find_path(self.grid, self.player.pos, goal)
        if path:
            self.player.set_path(path)
        return path

    def tick(self):
        now = time.monotonic()
        delta = now - self._last_time if self._last_time else self.FIXED_DT
        self._last_time = now
        self._accum += delta

        while self._accum >= self.FIXED_DT:
            self.player.update(self.input, self.grid, self.FIXED_DT)
            self._accum -= self.FIXED_DT
            self._tick += 1

    def run(self):
        self._running = True
        self._last_time = time.monotonic()
        while self._running:
            self.tick()

    def stop(self):
        self._running = False


def build_demo_level(grid: CollisionGrid):
    for x in range(GRID_WIDTH):
        grid.set_blocked(x, 0)
        grid.set_blocked(x, GRID_HEIGHT - 1)
    for y in range(GRID_HEIGHT):
        grid.set_blocked(0, y)
        grid.set_blocked(GRID_WIDTH - 1, y)

    _wall_segments = [
        (10, 5, 10, 20), (20, 10, 35, 10),
        (15, 30, 15, 50), (40, 20, 40, 45),
        (25, 50, 55, 50),
    ]

    for x0, y0, x1, y1 in _wall_segments:
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                grid.set_blocked(x, y)


if __name__ == "__main__":
    loop = GameLoop()
    build_demo_level(loop.grid)

    path = loop.request_path_to(50.0, 50.0)
    loop.tick()
    state = loop.player.get_state()
