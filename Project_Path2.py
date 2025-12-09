import numpy as np
import random
import time
from queue import Queue, LifoQueue
from heapq import heappush, heappop


def generate_random_maze(size):
    maze = np.random.choice([0, 1], size=(size, size), p=[0.6, 0.4])
    maze[0, 0] = maze[size-1, size-1] = 0
    return maze


def bfs(maze):
    start = (0, 0)
    goal = (len(maze) - 1, len(maze) - 1)
    queue = Queue()
    queue.put(start)
    visited = set([start])
    nodes_explored = 0

    while not queue.empty():
        node = queue.get()
        nodes_explored += 1
        if node == goal:
            return nodes_explored
        neighbors = get_neighbors(node, maze)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)

    return nodes_explored 


def dfs(maze):
    start = (0, 0)
    goal = (len(maze) - 1, len(maze) - 1)
    stack = LifoQueue()
    stack.put(start)
    visited = set([start])
    nodes_explored = 0

    while not stack.empty():
        node = stack.get()
        nodes_explored += 1
        if node == goal:
            return nodes_explored
        neighbors = get_neighbors(node, maze)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.put(neighbor)

    return nodes_explored



def ids(maze):
    for depth in range(0, len(maze)**2):
        result, nodes_explored = dls((0, 0), (len(maze) - 1, len(maze) - 1), maze, depth)
        if result:
            return nodes_explored
    return len(maze)**2

#------------------------------------------------------------------------------------------------------------------------------------------------------

def dls(node, goal, maze, depth):
    stack = [(node, 0)]
    visited = set()
    nodes_explored = 0

    while stack:
        (current_node, current_depth) = stack.pop()
        nodes_explored += 1
        if current_depth > depth:
            continue
        if current_node == goal:
            return True, nodes_explored
        visited.add(current_node)
        for neighbor in get_neighbors(current_node, maze):
            if neighbor not in visited:
                stack.append((neighbor, current_depth + 1))
    
    return False, nodes_explored


def a_star(maze):
    start = (0, 0)
    goal = (len(maze) - 1, len(maze) - 1)
    open_set = [(0, start)]
    g_costs = {start: 0}
    nodes_explored = 0

    while open_set:
        _, current = heappop(open_set)
        nodes_explored += 1
        if current == goal:
            return nodes_explored
        neighbors = get_neighbors(current, maze)
        for neighbor in neighbors:
            tentative_g_cost = g_costs[current] + 1
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                priority = tentative_g_cost + heuristic(neighbor, goal)
                heappush(open_set, (priority, neighbor))

    return nodes_explored


def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def get_neighbors(position, maze):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for direction in directions:
        neighbor = (position[0] + direction[0], position[1] + direction[1])
        if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze) and maze[neighbor[0]][neighbor[1]] == 0:
            neighbors.append(neighbor)
    return neighbors


def run_experiment(algorithm, num_trials=1000, size=10):
    total_time = 0
    total_nodes = 0
    for _ in range(num_trials):
        maze = generate_random_maze(size)
        start_time = time.time()
        nodes_explored = algorithm(maze)
        end_time = time.time()
        if nodes_explored is not None:
            total_nodes += nodes_explored
            total_time += end_time - start_time
    average_time = total_time / num_trials
    average_nodes = total_nodes // num_trials
    return average_time, average_nodes


def main():
    algorithms = {
        'BFS': bfs,
        'DFS': dfs,
        'IDS': ids,
        'A*': a_star
    }
    
    results = {}

    for name, algorithm in algorithms.items():
        average_time, average_nodes = run_experiment(algorithm)
        results[name] = (average_time, average_nodes)

    print(f"{'Algorithm':<10} | {'Average Time (s)':<15} | {'Average Nodes':<15}")
    print("-" * 45)
    for name, (avg_time, avg_nodes) in results.items():
        print(f"{name:<10} | {avg_time:<15.5f} | {avg_nodes:<15}")

if __name__ == "__main__":
    main()
