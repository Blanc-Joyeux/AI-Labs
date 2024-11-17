import pygame
from queue import Queue

# Initialize Pygame
pygame.init()

# Grid dimensions
ROWS, COLS = 20, 20
CELL_SIZE = 30
WINDOW_SIZE = (ROWS * CELL_SIZE, COLS * CELL_SIZE)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize screen
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Pathfinding Agent")

# Grid setup
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]

# More obstacles (positions)
obstacles = [
    (5, 5), (5, 6), (5, 7), (6, 5), (10, 10), (10, 11), (11, 10),
    (7, 14), (8, 14), (9, 14), (15, 5), (15, 6), (15, 7), (14, 7),
    (12, 18), (13, 18), (13, 19), (14, 18), (4, 2), (4, 3), (5, 2),
    (17, 9), (16, 9), (16, 10), (16, 11), (3, 17), (3, 18), (3, 19),
]

# Place obstacles in the grid
for obstacle in obstacles:
    grid[obstacle[0]][obstacle[1]] = -1  # Mark as obstacle

# Agent and Target positions
start = (0, 0)
target = (19, 19)
grid[target[0]][target[1]] = 2  # Mark target

# Draw the grid
def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[row][col] == -1:
                pygame.draw.rect(screen, BLACK, rect)
            elif grid[row][col] == 2:
                pygame.draw.rect(screen, RED, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLUE, rect, 1)  # Grid lines

# Pathfinding using BFS
def bfs(start, target):
    queue = Queue()
    queue.put((start, [start]))  # (current_position, path)
    visited = set()
    visited.add(start)

    while not queue.empty():
        current, path = queue.get()

        if current == target:
            return path

        # Possible moves (up, down, left, right)
        moves = [
            (current[0] - 1, current[1]),
            (current[0] + 1, current[1]),
            (current[0], current[1] - 1),
            (current[0], current[1] + 1),
        ]

        for move in moves:
            if (
                0 <= move[0] < ROWS
                and 0 <= move[1] < COLS
                and grid[move[0]][move[1]] != -1
                and move not in visited
            ):
                visited.add(move)
                queue.put((move, path + [move]))

    return None  # No path found

# Main loop
running = True
path = bfs(start, target)
agent_pos = start

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    draw_grid()

    # Draw the path
    if path:
        for pos in path:
            if pos != target:
                rect = pygame.Rect(pos[1] * CELL_SIZE, pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, YELLOW, rect)

    # Draw the agent
    rect = pygame.Rect(agent_pos[1] * CELL_SIZE, agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, rect)

    # Update agent's position along the path
    if path and agent_pos != target:
        agent_pos = path.pop(0)

    pygame.display.flip()
    pygame.time.delay(200)

pygame.quit()
