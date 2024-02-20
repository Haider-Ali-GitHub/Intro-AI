import heapq
from matplotlib import pyplot as plt
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

def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def a_star_search(maze, start, goal, tie_breaking='smaller_g'):
    # Heuristic function: Manhattan distance
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    # Initialize all the necessary dictionaries
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set_hash = {start}
    open_set = [(f_score[start], 0, start)]  # (f_score, g_score, position)
    came_from = {}
    
    while open_set:
        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path
        
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1]:
                if maze[neighbor[1], neighbor[0]] == 1:  # Check if wall
                    continue
                
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score:
                    g_score[neighbor] = np.inf
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        tie_break = -tentative_g_score if tie_breaking == 'smaller_g' else tentative_g_score
                        heapq.heappush(open_set, (f_score[neighbor], tie_break, neighbor))
                        open_set_hash.add(neighbor)
    
    return []  # Return an empty path if goal is not reachable


# Function to get valid neighbors of a cell
def get_neighbors(cell, maze):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for direction in directions:
        neighbor = (cell[0] + direction[0], cell[1] + direction[1])
        if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0]) and maze[neighbor[0]][neighbor[1]] == 0:
            neighbors.append(neighbor)
    return neighbors

# Function to reconstruct the path from start to goal
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# Generate one maze for demonstration
# maze_list = [generate_maze() for _ in range(50)]

# import matplotlib.pyplot as plt

# for i, maze in enumerate(maze_list):
#     plt.figure(figsize=(5, 5))
#     plt.imshow(maze, cmap='binary', origin='lower')
#     plt.title(f'Generated Maze {i+1}')
#     plt.show()

maze = generate_maze(10, 0.3)  # Assuming you have your generate_maze function
start = (0, 0)  # Starting and goal positions
goal = (maze.shape[0]-1, maze.shape[1]-1) 

path_smaller_g = a_star_search(maze, start, goal, tie_breaking='smaller_g')
print(f"Path with smaller g-values: {path_smaller_g}")

path_larger_g = a_star_search(maze, start, goal, tie_breaking='larger_g')
print(f"Path with larger g-values: {path_larger_g}")

plt.figure(figsize=(5, 5))
plt.imshow(maze, cmap='binary', origin='lower')
plt.title(f'Generated Maze')
plt.show()

# maze = maze_list[0]

# # Visualize the chosen maze
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 10))
# plt.imshow(maze, cmap='binary', origin='lower')
# plt.title('Generated Maze')
# plt.show()

