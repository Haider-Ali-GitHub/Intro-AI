import numpy as np
import time
import heapq
import random
import matplotlib.pyplot as plt


def generate_maze(size=101, p_blocked=0.3):
    grid = np.full((size, size), -1)  # Initialize all cells as unvisited
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Four directions (N, E, S, W)
    
    def is_valid(cell):
        x, y = cell
        return 0 <= x < size and 0 <= y < size and grid[y, x] == -1  # Check if cell is unvisited and within bounds
    
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
                grid[y, x] = 0  # Mark as free space
                neighbors = get_unvisited_neighbors(current_cell)
                if neighbors:
                    next_cell = random.choice(neighbors)
                    grid[next_cell[1], next_cell[0]] = 1 if random.random() < p_blocked else 0
                    if grid[next_cell[1], next_cell[0]] == 0:  # If next cell is free
                        stack.append(next_cell)  # Continue DFS
                else:
                    stack.pop()  # Backtrack if no unvisited neighbors
            else:
                stack.pop()  # Backtrack if current cell is already visited
    
    unvisited_cells = [(x, y) for x in range(size) for y in range(size)]
    while unvisited_cells:
        start_cell = random.choice(unvisited_cells)
        dfs(start_cell)
        unvisited_cells = [(x, y) for x, y in unvisited_cells if grid[y, x] == -1]
    
    return grid

# A* ALGORITHM USING BINARY HEAP

def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def a_star_search(maze, start, goal, tie_breaking):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set_hash = {start}
    expanded_cells = 0

    while open_set:
        current = heapq.heappop(open_set)[2]
        open_set_hash.remove(current)
        expanded_cells += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, expanded_cells

        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor[1], neighbor[0]] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        tie_break = -tentative_g_score if tie_breaking == 'larger_g' else tentative_g_score
                        heapq.heappush(open_set, (f_score[neighbor], tie_break, neighbor))
                        open_set_hash.add(neighbor)

    return [], expanded_cells

def repeated_a_star(maze, start, goal, tie_breaking='larger_g', search_direction='forward'):
    if search_direction == 'backward':
        start, goal = goal, start
    
    path, expanded_cells = a_star_search(maze, start, goal, tie_breaking)
    return path, expanded_cells



# MAZE PARAMETERS A* 
size = 101
p_blocked = 0.3
maze = generate_maze(size, p_blocked)
start = (0, 0)
goal = (size - 1, size - 1)

# FORWARD SEARCH RESULTS
path_forward, cells_expanded_forward = repeated_a_star(maze, start, goal, 'larger_g', 'forward')
print("Forward Search: Path Length =", len(path_forward), "Cells Expanded =", cells_expanded_forward)

# BACKWARD SEARCH RESULTS
path_backward, cells_expanded_backward = repeated_a_star(maze, start, goal, 'larger_g', 'backward')
print("Backward Search: Path Length =", len(path_backward), "Cells Expanded =", cells_expanded_backward)

# VISUALIZATION
plt.imshow(maze, cmap='binary')
plt.plot(*zip(*path_forward), marker='o', color='blue', linewidth=2, markersize=5, label='Forward Path')
plt.plot(*zip(*path_backward), marker='o', color='red', linewidth=2, markersize=5, label='Backward Path')
plt.legend()
plt.show()


# THE ULTIMATE COMPARISON!
def run_comparative_experiments(size=101, p_blocked=0.3, num_mazes=10):
    forward_times = []
    backward_times = []
    forward_expanded = []
    backward_expanded = []
    
    for i in range(num_mazes):
        print(f"Running maze {i+1}/{num_mazes}...")
        maze_start_time = time.time()  # Time start for generating each maze
        maze = generate_maze(size, p_blocked)
        print(f"Maze {i+1} generated in {time.time() - maze_start_time:.5f} seconds.")

        start, goal = (0, 0), (size - 1, size - 1)
        
        # Forward search
        start_time = time.time()
        path_forward, cells_expanded_forward = repeated_a_star(maze, start, goal, 'larger_g', 'forward')
        forward_duration = time.time() - start_time
        forward_times.append(forward_duration)
        forward_expanded.append(cells_expanded_forward)
        
        # Backward search
        start_time = time.time()
        path_backward, cells_expanded_backward = repeated_a_star(maze, start, goal, 'larger_g', 'backward')
        backward_duration = time.time() - start_time
        backward_times.append(backward_duration)
        backward_expanded.append(cells_expanded_backward)
        
        # PRINT RESULTS FOR EACH MAZE
        print(f"Maze {i+1} Forward Search: Time = {forward_duration:.5f}s, Expanded Cells = {cells_expanded_forward}")
        print(f"Maze {i+1} Backward Search: Time = {backward_duration:.5f}s, Expanded Cells = {cells_expanded_backward}")
    
    # PRINT AVERAGE 
    print("\n--- Summary of Results ---")
    print("Forward Search - Average Time: {:.5f}s, Average Expanded Cells: {:.2f}".format(np.mean(forward_times), np.mean(forward_expanded)))
    print("Backward Search - Average Time: {:.5f}s, Average Expanded Cells: {:.2f}".format(np.mean(backward_times), np.mean(backward_expanded)))

# RUN THE ANALYSIS
size_of_grid = 101
mazes_to_test = 10

run_comparative_experiments(size=101, p_blocked=0.3, num_mazes=50)


