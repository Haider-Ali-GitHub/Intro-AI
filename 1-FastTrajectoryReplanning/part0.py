import heapq
import numpy as np
import random

def generate_maze(size=101, p_blocked=0.3):
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

def a_star_search(grid, start, goal, tie_breaking):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0 if tie_breaking == "g-min" else -0, start))
    closed_set = set()
    came_from = {}

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        closed_set.add(current)

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]) or grid[neighbor[1]][neighbor[0]] == 1:
                continue

            if neighbor in closed_set:
                continue

            tentative_g = g + 1 if tie_breaking == "g-min" else -(g + 1)
            if tie_breaking == "g-min":
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))
            else:  # For "g-max", we push with negative g to prioritize larger g-values
                heapq.heappush(open_set, (-(tentative_g - heuristic(neighbor, goal)), tentative_g, neighbor))
            came_from[neighbor] = current

    return None


# Generate one maze for demonstration
maze = generate_maze()
start, goal = (0, 0), (100, 100)  # Example start and goal, adjust as needed
path = a_star_search(maze, start, goal, tie_breaking="g-min")  # Use "g-min" or "g-max" for tie_breaking

# Visualize the maze
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(maze, cmap='binary', origin='lower') 
plt.title('Generated Maze')
plt.show()

