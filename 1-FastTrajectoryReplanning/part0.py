import numpy as np
import random

def generate_maze(size=10, p_blocked=0.3):
    # Initialize the gridworld as unvisited (-1), 0 is unblocked, 1 is blocked
    grid = np.full((size, size), -1)
    
    # Directions (up, right, down, left)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    def is_valid(cell):
        x, y = cell
        return 0 <= x < size and 0 <= y < size and grid[y, x] == -1
    
    def get_unvisited_neighbors(cell):
        x, y = cell
        neighbors = [(x + dx, y + dy) for dx, dy in directions]
        return [n for n in neighbors if is_valid(n)]
    
    def dfs(cell):
        stack = [cell]
        while stack:
            current_cell = stack[-1]
            x, y = current_cell
            if grid[y, x] == -1:  # If current cell is unvisited
                grid[y, x] = 0  # Mark as visited and unblocked
                neighbors = get_unvisited_neighbors(current_cell)
                if neighbors:
                    next_cell = random.choice(neighbors)
                    if random.random() < p_blocked:
                        grid[next_cell[1], next_cell[0]] = 1  # Blocked
                    else:
                        stack.append(next_cell)  # Unblocked and to be visited
                else:
                    stack.pop()  # Backtrack
            else:
                stack.pop()  # Backtrack
    
    # Start DFS from a random cell until all cells are visited
    unvisited_cells = [(x, y) for x in range(size) for y in range(size)]
    while unvisited_cells:
        start_cell = random.choice(unvisited_cells)
        dfs(start_cell)
        unvisited_cells = [(x, y) for x, y in unvisited_cells if grid[y, x] == -1]

    return grid



# Generate one maze for demonstration
maze_list = [generate_maze() for _ in range(50)]

import matplotlib.pyplot as plt

for i, maze in enumerate(maze_list):
    plt.figure(figsize=(5, 5))
    plt.imshow(maze, cmap='binary', origin='lower')
    plt.title(f'Generated Maze {i+1}')
    plt.show()

# maze = maze_list[0]

# # Visualize the chosen maze
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# plt.imshow(maze, cmap='binary', origin='lower')
# plt.title('Generated Maze')
# plt.show()



