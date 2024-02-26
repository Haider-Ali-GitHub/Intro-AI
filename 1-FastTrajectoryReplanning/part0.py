import heapq
from matplotlib import pyplot as plt
import numpy as np
import random

def generate_maze(size=10, p_blocked=0.3):
    grid = np.full((size, size), -1)
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
            if grid[y, x] == -1:
                grid[y, x] = 0
                neighbors = get_unvisited_neighbors(current_cell)
                if neighbors:
                    next_cell = random.choice(neighbors)
                    grid[next_cell[1], next_cell[0]] = 1 if random.random() < p_blocked else 0
                    if grid[next_cell[1], next_cell[0]] == 0:  # Continue DFS only for unblocked
                        stack.append(next_cell)
                else:
                    stack.pop()
            else:
                stack.pop()
    
    unvisited_cells = [(x, y) for x in range(size) for y in range(size)]
    while unvisited_cells:
        start_cell = random.choice(unvisited_cells)
        dfs(start_cell)
        unvisited_cells = [(x, y) for x, y in unvisited_cells if grid[y, x] == -1]
    
    return grid

num_mazes = 50  #Adjust to 50 if you want to generate 50 
maze_size = 101
p_blocked = 0.3


mazes = [generate_maze(maze_size, p_blocked) for _ in range(num_mazes)]

# plt.figure(figsize=(5, 5))
# plt.imshow(mazes[0], cmap='binary', origin='lower')
# plt.title(f'Maze 1 of {num_mazes}')
# plt.show()

for i, maze in enumerate(mazes):
    plt.figure(figsize=(5, 5))
    plt.imshow(maze, cmap='binary', origin='lower')
    plt.title(f'Maze {i+1} of {len(mazes)}')
    plt.show()

# size = 10
# maze = generate_maze(size, 0.3)
# start = (0, 0)
# goal = (size - 1, size - 1)

# path_smaller_g, cells_expanded_smaller_g = repeated_forward_a_star(maze, start, goal, 'smaller_g')
# print(f"Path with smaller g-values: {path_smaller_g}")
# print(f"Cells expanded with smaller g-values: {cells_expanded_smaller_g}")

# path_larger_g, cells_expanded_larger_g = repeated_forward_a_star(maze, start, goal, 'larger_g')
# print(f"Path with larger g-values: {path_larger_g}")
# print(f"Cells expanded with larger g-values: {cells_expanded_larger_g}")

# plt.figure(figsize=(5, 5))
# plt.imshow(maze, cmap='binary', origin='lower')
# plt.title(f'Generated Maze')
# plt.show()
