import heapq
from matplotlib import pyplot as plt
import numpy as np
import random
import time

def generate_maze(size=101, p_blocked=0.3):
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

def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def adaptive_a_star(maze, start, goal):
    h_values = {}
    g_scores = {start: 0}

    def heuristic(node, goal):
        # Access h_values and g_scores directly since they are now in the enclosing scope
        g_goal = g_scores.get(goal, 0)  # Assuming g_goal is the g_score of the goal
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1]) if node not in h_values else g_goal - g_scores[node]

    def get_neighbors(node):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = []
        for d in directions:
            neighbor = (node[0] + d[0], node[1] + d[1])
            if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0]) and maze[neighbor[1]][neighbor[0]] == 0:
                neighbors.append(neighbor)
        return neighbors

    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start))  # Adjusted call to heuristic
    came_from = {}
    closed_set = set()

    while open_list:
        current = heapq.heappop(open_list)[2]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            # Update heuristics for all visited nodes
            for node in closed_set:
                h_values[node] = g_scores[goal] - g_scores[node]
            return path, len(closed_set)

        closed_set.add(current)

        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue
            tentative_g_score = g_scores[current] + 1

            if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                came_from[neighbor] = current
                g_scores[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)  # Adjusted call to heuristic
                heapq.heappush(open_list, (f_score, tentative_g_score, neighbor))

    return [], 0  # Return an empty path and 0 if the goal is not reachable


def a_star_search(maze, start, goal, tie_breaking):
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    open_set_hash = {start}
    open_set = [(f_score[start], 0, start)]
    came_from = {}
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
        
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor[1], neighbor[0]] != 1:
                tentative_g_score = g_score.get(neighbor, np.inf) + 1
                if tentative_g_score < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        tie_break = -tentative_g_score if tie_breaking == 'smaller_g' else tentative_g_score
                        heapq.heappush(open_set, (f_score[neighbor], tie_break, neighbor))
                        open_set_hash.add(neighbor)
    
    return [], expanded_cells

def repeated_forward_a_star(maze, start, goal, tie_breaking):
    discovered_maze = np.full_like(maze, -1)  # All cells are unknown
    discovered_maze[start[1], start[0]] = 0  # Start cell is free
    path = [start]
    current = start
    expanded_cells = 0

    while current != goal:
        open_set = []
        g_score = {current: 0}
        f_score = {current: heuristic(current, goal)}
        open_set_hash = {current}
        came_from = {}
        
        while open_set_hash:
            current = min(open_set_hash, key=lambda x: (f_score[x], g_score[x] if tie_breaking == 'larger_g' else -g_score[x]))
            if current == goal:
                total_path = []
                while current in came_from:
                    total_path.append(current)
                    current = came_from[current]
                return total_path[::-1], expanded_cells
            
            open_set_hash.remove(current)
            expanded_cells += 1
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1]:
                    if discovered_maze[neighbor[1], neighbor[0]] == -1:
                        discovered_maze[neighbor[1], neighbor[0]] = maze[neighbor[1], neighbor[0]]
                    
                    if discovered_maze[neighbor[1], neighbor[0]] == 0:
                        tentative_g_score = g_score[current] + 1
                        if tentative_g_score < g_score.get(neighbor, np.inf):
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                            if neighbor not in open_set_hash:
                                open_set_hash.add(neighbor)
    
        # No path found
        return [], expanded_cells



def run_experiments(size=101, p_blocked=0.3, num_mazes=50):
    total_runtime_rf = 0
    total_runtime_adaptive = 0
    
    for i in range(num_mazes):
        maze = generate_maze(size, p_blocked)
        start = (0, 0)
        goal = (size - 1, size - 1)

        # Measure Repeated Forward A* runtime
        start_time = time.time()
        path_rf, expanded_rf = repeated_forward_a_star(maze, start, goal, 'larger_g')
        end_time = time.time()
        runtime_rf = end_time - start_time
        total_runtime_rf += runtime_rf

        # Measure Adaptive A* runtime
        start_time = time.time()
        path_ad, expanded_ad = adaptive_a_star(maze, start, goal)
        end_time = time.time()
        runtime_adaptive = end_time - start_time
        total_runtime_adaptive += runtime_adaptive

        print(f"Maze {i+1} Repeated Forward A* Path Length: {len(path_rf)}, Cells Expanded: {expanded_rf}, Runtime: {runtime_rf} seconds")
        print(f"Maze {i+1} Adaptive A* Path Length: {len(path_ad)}, Cells Expanded: {expanded_ad}, Runtime: {runtime_adaptive} seconds")

    # Calculate average runtimes
    avg_runtime_rf = total_runtime_rf / num_mazes
    avg_runtime_adaptive = total_runtime_adaptive / num_mazes

    print(f"\nAverage Runtime for Repeated Forward A*: {avg_runtime_rf} seconds")
    print(f"Average Runtime for Adaptive A*: {avg_runtime_adaptive} seconds")



run_experiments()




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
