import numpy as np
import random
import heapq
import matplotlib.pyplot as plt

class Node:
    def __init__(self, position, g=float('inf'), h=0, f=None, parent=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = f if f is not None else self.g + self.h
        self.parent = parent

    def __lt__(self, other):
        return (self.f, -self.g) < (other.f, -other.g)

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
                    if random.random() < p_blocked:
                        grid[next_cell[1], next_cell[0]] = 1
                    else:
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
    
    grid[0, 0] = 0  # Ensure start is unblocked
    grid[size-1, size-1] = 0  # Ensure goal is unblocked
    return grid

def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def get_neighbors(node, maze):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    neighbors = []
    for direction in directions:
        next_position = (node.position[0] + direction[0], node.position[1] + direction[1])
        if 0 <= next_position[0] < maze.shape[0] and 0 <= next_position[1] < maze.shape[1]:
            if maze[next_position[1], next_position[0]] == 0:
                neighbors.append(Node(next_position))
    return neighbors

def reconstruct_path(current_node):
    path = []
    while current_node:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1]

def a_star_search(maze, start, goal):
    start_node = Node(start, g=0, h=manhattan_distance(start, goal))
    goal_node = Node(goal)
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    closed_set = set()

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        closed_set.add(current_node.position)

        if current_node.position == goal:
            return reconstruct_path(current_node)

        for neighbor in get_neighbors(current_node, maze):
            if neighbor.position in closed_set or maze[neighbor.position[1], neighbor.position[0]] == 1:
                continue
            tentative_g_score = current_node.g + 1

            if tentative_g_score < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g_score
                neighbor.h = manhattan_distance(neighbor.position, goal)
                neighbor.f = neighbor.g + neighbor.h

                if not any(node[1].position == neighbor.position for node in open_list):
                    heapq.heappush(open_list, (neighbor.f, neighbor))

    return None

def visualize_path(maze, mazeNum , path, start, goal):
    plt.figure(figsize=(10, 10))
    # Display the maze
    plt.imshow(maze, cmap='binary', origin='lower')

    # Extract path positions for plotting
    path_x, path_y = zip(*path)

    # Mark the path
    plt.plot(path_x, path_y, color="blue", linewidth=3, linestyle='-', label='Shortest Path')

    # Mark the start and goal with distinct colors
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')  # Green dot for start
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')  # Red dot for goal

    plt.title(f'Maze {mazeNum}')
    plt.legend()
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()



# Generate mazes and visualize paths
# Assuming the previous definitions for maze generation and A* search are correct

# Generate mazes
maze_list = [generate_maze(size=10, p_blocked=0.3) for _ in range(5)]  # Example with 5 mazes for brevity

for i, maze in enumerate(maze_list[:5]):  # Example: Process the first 5 mazes
    start = (0, 0)
    goal = (maze.shape[0]-1, maze.shape[1]-1)
    path = a_star_search(maze, start, goal)
    if path:
        visualize_path(maze, i+1, path, start, goal)
    else:
        print(f"No path found for Maze {i+1}")


