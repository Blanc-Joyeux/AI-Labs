import pygame
import sys
from collections import deque
import time

# Constants for the grid
GRID_SIZE = 30  # Size of the grid cells (pixels)
WIDTH, HEIGHT = 20, 20  # Width and height of the grid (20x20 grid)
WINDOW_SIZE = (WIDTH * GRID_SIZE, HEIGHT * GRID_SIZE)  # Window size

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Initialize the Pygame window
pygame.init()
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Pathfinding Agent")

# Define the start and target positions
start = (0, 0)
target = (19, 19)

# Define a list of obstacles (represented as grid coordinates)
obstacles = [
    (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),  # Vertical wall
    (10, 3), (10, 4), (10, 5),  # Horizontal wall
    (15, 10), (15, 11), (15, 12),  # Another vertical wall
    (7, 14), (8, 14), (9, 14),  # Horizontal set
    (3, 17), (4, 17), (5, 17),  # Obstacles near the target
    (12, 6), (12, 7), (12, 8),  # Random block of obstacles
    (17, 1), (18, 1), (19, 1),  # Some more obstacles
    (13, 14), (14, 14), (13, 15), (14, 15),  # Complex block
    (16, 17), (16, 18), (17, 17), (17, 18),  # Another block near the end
]


# Function to draw the grid and obstacles
def draw_grid():
    window.fill(WHITE)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(window, BLUE, rect, 1)
    # Draw the obstacles
    for obstacle in obstacles:
        x, y = obstacle
        pygame.draw.rect(window, BLACK, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))


# Function to check if a cell is passable
def is_passable(cell):
    return cell not in obstacles and 0 <= cell[0] < WIDTH and 0 <= cell[1] < HEIGHT


# Breadth-First Search (BFS) for pathfinding
def bfs(start, target):
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()

        if current == target:
            break

        neighbors = [(current[0] + 1, current[1]), (current[0] - 1, current[1]),
                     (current[0], current[1] + 1), (current[0], current[1] - 1)]

        for neighbor in neighbors:
            if neighbor not in came_from and is_passable(neighbor):
                queue.append(neighbor)
                came_from[neighbor] = current

    # Reconstruct the path
    path = []
    if target in came_from:
        current = target
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()
    return path


# Main loop
running = True
path = bfs(start, target)  # Get the path before starting the animation
current_position = start
visited_cells = []  # Store cells visited by the agent

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw the grid and obstacles
    draw_grid()

    # Draw the cells the agent has visited in yellow
    for cell in visited_cells:
        pygame.draw.rect(window, YELLOW, (cell[0] * GRID_SIZE, cell[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw the agent and target
    pygame.draw.rect(window, GREEN,
                     (current_position[0] * GRID_SIZE, current_position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Agent
    pygame.draw.rect(window, RED, (target[0] * GRID_SIZE, target[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Target

    # Update the display
    pygame.display.flip()

    # Move the agent along the path (animate step by step)
    if current_position != target:
        next_position = path.pop(0)  # Get the next step in the path
        visited_cells.append(current_position)  # Mark the current position as visited
        current_position = next_position  # Move the agent to the next position
        time.sleep(0.1)  # Delay for animation effect

pygame.quit()
sys.exit()
