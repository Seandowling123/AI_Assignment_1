from pyamaze import maze, agent, COLOR, textTitle
from collections import Counter
import random
import math
import time
import os

# Some global variables
maze_speed = 0
value_iteration_values = 0
policy_iteration_values = 0
policy_iteration_path = 0
queued_cells = []
num_iterations = 0

# Find the saved maze csv file
def find_maze_file():
    files = os.listdir('.')
    csv_files = [file for file in files if file.endswith('.csv')]
    if csv_files:
        return csv_files[0]
    else:
        return None

# Delete all maze files. For the end of the program
def delete_all_maze_files():
    files = os.listdir('.')
    csv_files = [file for file in files if file.endswith('.csv') and file[0:4] == 'maze']
    for csv_file in csv_files:
        os.remove(csv_file)

# Print performance metrics for the algoritms
def print_metrics(algo_name, path_length, nodes_searched, num_cells_queued, iterations, elapsed_time):
    print(f"{algo_name} Metrics:")
    print(f"  Path Length:   \t{path_length:}")
    print(f"  Nodes Searched:\t{nodes_searched:}")
    print(f"  Nodes Queued:  \t{num_cells_queued:}")
    print(f"  Iterations:    \t{iterations:}")
    print(f"  Elapsed Time:  \t{elapsed_time:} seconds\n")
        
# Display the maze being solved
def run_algorithm(maze, start, goal, search_func):
    # Define the paths
    start_time = time.time()
    algo_name, path, search = search_func(maze, start, goal)
    end_time = time.time()
    
    # Get performance metrics
    nodes_searched = 'N/A'
    num_cells_queued = 'N/A'
    path_length = len(path)
    global num_iterations
    iterations = num_iterations
    num_iterations = 0
    elapsed_time = end_time - start_time
    if algo_name != "Value Iteration" and algo_name != "Policy Iteration":
        nodes_searched = len(search)
        global queued_cells
        num_cells_queued = sum(1 for count in Counter(map(tuple, queued_cells)).values() if count == 1)
        queued_cells = []
    metrics = [str(path_length), str(nodes_searched), str(num_cells_queued), iterations, "{:.4f}".format(elapsed_time)]
    print_metrics(algo_name, *metrics)
    
    # Define maze-solving agents
    search_agent = agent(maze,start[0],start[1],filled=True,footprints=True,color=COLOR.cyan,name=algo_name,metrics=metrics)
    solve_agent = agent(maze,start[0],start[1],filled=True,footprints=True,color=COLOR.green,name=algo_name,metrics=metrics)
    
    # Set the speed for the agent 
    global maze_speed
    
    # Display maze search and solve
    if algo_name != "Value Iteration" and algo_name != "Policy Iteration":
        maze.tracePath({search_agent:search}, delay=1, kill=True)
    elif algo_name == "Value Iteration":
        global value_iteration_values
        value_iteration_values = search
    elif algo_name == "Policy Iteration":
        global policy_iteration_values
        policy_iteration_values = search
    maze.tracePath({solve_agent:path}, delay=maze_speed, kill=True)
    
# Converts a path of cells to a string of directions
def to_directions(path):
    directions = ''
    for i in range(1, len(path)):
        current_y, current_x = path[i-1]
        next_y, next_x = path[i]

        if next_x > current_x:
            directions += 'E'
        elif next_x < current_x:
            directions += 'W'
        if next_y > current_y:
            directions += 'S'
        elif next_y < current_y:
            directions += 'N'
    return directions
    
# Return a list of the available adjacent cells
def get_available_cells(maze_map, current_pos, checked_cells):
    available_positions = []
    if maze_map[current_pos]['N']:
        new_pos = (current_pos[0]-1, current_pos[1])
        if new_pos not in checked_cells:
            available_positions.append(new_pos)
    if maze_map[current_pos]['S']:
        new_pos = (current_pos[0]+1, current_pos[1])
        if new_pos not in checked_cells:
            available_positions.append(new_pos)
    if maze_map[current_pos]['E']:
        new_pos = (current_pos[0], current_pos[1]+1)
        if new_pos not in checked_cells:
            available_positions.append(new_pos)
    if maze_map[current_pos]['W']:
        new_pos = (current_pos[0], current_pos[1]-1)
        if new_pos not in checked_cells:
            available_positions.append(new_pos)
    return available_positions

# Return a list of the adjacent cells
def get_neighbouring_cells(maze_map, current_pos):
    available_positions = []
    if maze_map[current_pos]['N']:
        new_pos = (current_pos[0]-1, current_pos[1])
        available_positions.append(new_pos)
    if maze_map[current_pos]['S']:
        new_pos = (current_pos[0]+1, current_pos[1])
        available_positions.append(new_pos)
    if maze_map[current_pos]['E']:
        new_pos = (current_pos[0], current_pos[1]+1)
        available_positions.append(new_pos)
    if maze_map[current_pos]['W']:
        new_pos = (current_pos[0], current_pos[1]-1)
        available_positions.append(new_pos)
    return available_positions

def monitor_que(current_que):
    global queued_cells
    queued_cells.append(current_que)

# -----------------
# Search Algorithms
# -----------------

# Solve the maze with BFS
def BFS(maze, start, goal):
    maze_map = maze.maze_map
    que = []
    checked_cells = []
    prev_pos = {}
    
    # Initialise variables
    current_pos = start
    past_cells = [current_pos]
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    iterations = 0
    
    # While the goal has not been reached & the que is not empty
    while((current_pos != goal) and (len(que) > 0)):
        iterations = iterations+1
        
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        que = que + available_cells
        monitor_que(que)
        for cell in available_cells:
            prev_pos[cell] = current_pos
        
        # Get next unchecked cell from the que
        while que[0] in checked_cells:
            del que[0]
        current_pos = que[0]
        past_cells.append(current_pos)
        
        if current_pos in checked_cells:
            print("shit")
    
    global num_iterations
    num_iterations = iterations
    
    # If the goal has been reached, get the path
    if current_pos == goal:    
        path = []
        path.append(current_pos)
        while current_pos != start:
            prev = prev_pos[current_pos]
            path.append(prev)
            current_pos = prev
    
    # Convert the path to a set of directions
    path_reversed = path[::-1]
    directions = to_directions(path_reversed)
    return "BFS", directions, past_cells

# Solve the maze with DFS
def DFS(maze, start, goal):
    maze_map = maze.maze_map
    que = []
    checked_cells = []
    prev_pos = {}
    
    # Initialise variables
    current_pos = start
    past_cells = [current_pos]
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    iterations = 0
    
    # While the goal has not been reached & the que is not empty
    while((current_pos != goal) and (len(que) > 0)):
        iterations = iterations+1
        
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        que = que + available_cells
        monitor_que(que)
        for cell in available_cells:
            prev_pos[cell] = current_pos
        
        # Get next unchecked cell from the que
        while que[len(que)-1] in checked_cells:
            del que[len(que)-1]
        current_pos = que[len(que)-1]
        del que[len(que)-1]
        past_cells.append(current_pos)
        
    global num_iterations
    num_iterations = iterations
        
    # If the goal has been reached, get the path
    if current_pos == goal:    
        path = []
        path.append(current_pos)
        while current_pos != start:
            prev = prev_pos[current_pos]
            path.append(prev)
            current_pos = prev
    
    # Convert the path to a set of directions
    path_reversed = path[::-1]
    directions = to_directions(path_reversed)
    return "DFS", directions, past_cells

# Calculates Manhattan distance between a cell and the goal
def get_heuristic(cell, goal):
    distance = abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
    return distance

# Calculates the g value for each available cell 
def get_g(available_cells, current_pos, g_values):
    costs = {}
    for cell in available_cells:
        cost = g_values[current_pos]+1
        costs[cell] = cost
    return costs

# Calculate the total cost for each available cell   
def get_cell_costs(available_cells, g_values, current_costs, goal):
    costs = {}
    for cell in available_cells:
        cost = g_values[cell] + get_heuristic(cell, goal)
        costs[cell] = cost
    return costs

# Return the cell with the lowest associated cost
def get_lowest_cost_cell(cell_costs):
    min_key = min(cell_costs, key=cell_costs.get)
    return min_key

# Solve the maze with A*
def A_star(maze, start, goal):
    maze_map = maze.maze_map
    checked_cells = []
    prev_pos = {}
    g_values = {}
    cell_costs = {}
    
    # Initialise variables
    current_pos = start
    g_values[current_pos] = 0
    cell_costs[current_pos] = get_heuristic(current_pos, goal)
    past_cells = [current_pos]
    available_cells = get_available_cells(maze_map, current_pos, checked_cells)
    g_values.update(get_g(available_cells, current_pos, g_values))
    cell_costs.update(get_cell_costs(available_cells, g_values, cell_costs, goal))
    iterations = 0
    
    # While the goal has not been reached & the que is not empty
    while(current_pos != goal and (len(cell_costs) > 0)):
        iterations = iterations+1
        
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        
        # Calculate heuristics
        g_values.update(get_g(available_cells, current_pos, g_values))
        cell_costs.update(get_cell_costs(available_cells, g_values, cell_costs, goal))
        monitor_que(list(cell_costs.keys()))
        
        # Set previous postions
        for cell in available_cells:
            prev_pos[cell] = current_pos
    
        # Get next unchecked cell from the que
        while get_lowest_cost_cell(cell_costs) in checked_cells:
            del cell_costs[get_lowest_cost_cell(cell_costs)]
        current_pos = get_lowest_cost_cell(cell_costs)
        del cell_costs[current_pos]
        past_cells.append(current_pos)
        
    global num_iterations
    num_iterations = iterations
        
    # If the goal has been reached, get the path
    if current_pos == goal:    
        path = []
        path.append(current_pos)
        while current_pos != start:
            prev = prev_pos[current_pos]
            path.append(prev)
            current_pos = prev
    
    # Convert the path to a set of directions
    path_reversed = path[::-1]
    directions = to_directions(path_reversed)
    return "A*", directions, past_cells

# --------------
# MDP Algorithms
# --------------

# Creates a new maze
def create_maze(starter_maze, name):
    maze_file = find_maze_file()
    new_maze=maze(starter_maze.rows,starter_maze.cols,name=name)
    new_maze.CreateMaze(goal[0],goal[1],loopPercent=30,theme="dark", loadMaze=maze_file)
    textTitle(new_maze, "", "")
    return new_maze

# Display the value iteration values on the maze
def show_cell_values(m, values):
    maze = create_maze(m, "Value Iteration\nSolution")
    
    cell_agents = []
    i = 0
    print("Loading Solution Maze:")
    for value in values:
        i = i+1
        if i % (len(values)/5) == 0:
            print(f"{(i/(len(values)))*100}%")
        cell_agent = agent(maze,value[0],value[1],filled=True,footprints=False,color=COLOR.from_value(values[value]),name="Value Iteration")
        cell_agents.append(cell_agent)
        
    for cell_agent in cell_agents:
        maze.tracePath({cell_agent:[(1,1)]}, delay=1, kill=False)
    maze.run()

# Return the cell with the highest associated value
def get_highest_value_cell(cells, V):
    max_cell = None
    max_value = float('-inf')
    for cell in cells:
        if V[cell] > max_value:
            max_value = V[cell]
            max_cell = cell
    return max_cell

# Get the lowest cost path through the maze from the values
def get_value_iteration_path(maze_map, V, start, goal):
    # Initialise variables
    current_pos =  start
    path = []
    path.append(current_pos)
    # Travel along the path of highest value cells
    while current_pos != goal:
        #print(current_pos)
        neighbouring_cells = get_neighbouring_cells(maze_map, current_pos)
        highest_value_cell = get_highest_value_cell(neighbouring_cells, V)
        current_pos = highest_value_cell
        path.append(current_pos)
    return path
        
# Calculate the value for a cell with Bellman eq
def bellman_eq(current_pos, neighbouring_cells, R, V, discount):
    V_values = []
    # Reward for making to the goal
    if current_pos in R:
        cell_R = R[current_pos]
        V_values.append(cell_R)
    else:
        # Discounted reward
        for cell in neighbouring_cells:
            cell_V = 0
            if cell in V:
                cell_V = V[cell]
            V_values.append(discount*cell_V)
    return max(V_values)

# Solve the maze with value iteration
def value_iteration(maze, start, goal):
    maze_map = maze.maze_map
    discount = .9
    R = {}
    V = {}
    old_V = 0
    delta = 0
    deltas = []
    R[goal] = 1
    delta = 1
    iterations = 0
    
    # Calculate the values for each cell in the maze until convergence
    while delta > 0:
        iterations = iterations+1
        deltas = []
        old_V = V.copy()
        
        # Update the value of each cell
        for cell in maze_map:
            neighbouring_cells = get_neighbouring_cells(maze_map, cell)
            V[cell] = bellman_eq(cell, neighbouring_cells, R, old_V, discount)
            
            # Calculate convergence
            if cell in old_V:
                cell_delta = V[cell] - old_V[cell]
            else: cell_delta = V[cell]
            deltas.append(cell_delta)
        delta = abs(sum(deltas)/len(V))
        
    global num_iterations
    num_iterations = iterations
    
    # Get path through the maze
    path = get_value_iteration_path(maze_map, V, start, goal)
    directions = to_directions(path)
    return "Value Iteration", directions, V

# Generate a random policy for each cell
def initialise_policy(maze_map):
    policy = {}
    directions = ["N", "S", "E", "W"]
    for cell in maze_map:
        rand_direction = directions[random.randint(0,3)]
        policy[cell] = rand_direction
    return policy

# Get the coordinates of the cell in a given direction
def get_cell_from_direction(cell, direction):
    new_cell = (-1,-1)
    if direction == 'E':
        new_cell = (cell[0], cell[1]+1)
    elif direction == 'W':
        new_cell = (cell[0], cell[1]-1)
    elif direction == 'S':
        new_cell = (cell[0]+1, cell[1])
    elif direction == 'N':
        new_cell = (cell[0]-1, cell[1])
    return new_cell
    
# Calculate the value for each cell
def policy_evaluation(maze_map, policy, R, old_V, discount, goal):
    V = {}
    for cell in maze_map:
        cell_V = 0
        neighbouring_cells = get_neighbouring_cells(maze_map, cell)
        
        # Special case for the goal
        if cell == goal:
            cell_V = R[cell]
        
        # Update the value
        elif get_cell_from_direction(cell, policy[cell]) in neighbouring_cells:
            cell_V = bellman_eq(cell, [get_cell_from_direction(cell, policy[cell])], R, old_V, discount)
        V[cell] = cell_V
    return V

# Find the optimal policy for each cell
def get_optimal_policy(cell, neighbouring_cells, policy, V):
    highest_V_cell = get_cell_from_direction(cell,policy[cell])
    highest_V = 0
    
    # Set policy to go to the highest value cell
    for neigbouring_cell in neighbouring_cells:
        if neigbouring_cell in V:
            if V[neigbouring_cell] > highest_V:
                highest_V = V[neigbouring_cell]
                highest_V_cell = neigbouring_cell
    direction = to_directions([cell, highest_V_cell])
    return direction

# Update the policy of each cell to maximise value
def policy_improvement(maze_map, policy, V):
    new_policy = {}
    for cell in maze_map:
        neighbouring_cells = get_neighbouring_cells(maze_map, cell)
        new_policy[cell] = get_optimal_policy(cell, neighbouring_cells, policy, V)
    return new_policy

# Check if two dicts are equal
def dicts_equal(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1.keys():
        if dict1[key] != dict2[key]:
            return False
    return True

# Get the solve the maze from the cell values
def get_policy_iteration_path(start, goal, policy):
    current_pos =  start
    path = []
    path.append(current_pos)
    
    # Travel along the path of cell policies
    while current_pos != goal:
        next_cell = get_cell_from_direction(current_pos, policy[current_pos])
        current_pos = next_cell
        path.append(current_pos)
    return path
        
# Solve the maze with policy iteration
def policy_iteration(maze, start, goal):
    maze_map = maze.maze_map
    discount = .9
    R = {}
    V = {}
    R[goal] = 1
    policy = initialise_policy(maze_map)
    policy_unchanged = 0
    iterations = 0
    
    # Update the policy until convergence
    while not policy_unchanged or 0 in V.values():
        iterations = iterations+1
        old_V = V.copy()
        old_policy = policy.copy()
        
        V = policy_evaluation(maze_map, policy, R, old_V, discount, goal)
        policy = policy_improvement(maze_map, policy, V)
        
        policy_unchanged = dicts_equal(policy, old_policy)
    
    global num_iterations
    num_iterations = iterations
    
    # Get path through the maze
    path = get_policy_iteration_path(start, goal, policy)
    global policy_iteration_path
    policy_iteration_path = path
    directions = to_directions(path)
    return "Policy Iteration", directions, policy

# Convert directions list to orientations for display
def directions_to_orientations(directions):
    orientation_conversion = {'N': 0, 'S': 2, 'E': 1, 'W': 3}
    orientations = []
    
    for direction in directions:
        orientations.append(orientation_conversion[direction])
    return orientations

def show_policy(m, policy):
    maze = create_maze(m, "Policy Iteration\nSolution")
    
    # Create an agent for each arrow
    cell_agents = []
    for cell in policy:
        if cell == (1,1):
            colour = COLOR.green
        elif cell in policy_iteration_path:
            colour = COLOR.chartreuse
        else: colour = COLOR.cadetblue
        cell_agent = agent(maze,cell[0],cell[1],shape='arrow',footprints=False,orient=directions_to_orientations(policy[cell])[0],color=colour,name="Policy Iteration")
        cell_agents.append(cell_agent)
    
    # Plot the agents
    for cell_agent in cell_agents:
        maze.tracePath({cell_agent:[(0,0)]}, delay=1, kill=False)
    maze.run()
    
def maze_speed_eq(size):
    return 150*math.e**(-0.0071*size)
    
# Start procedure
maze_options = {'S': '5x5','M': '25x25','L': '50x50'}
user_input = 0
while user_input not in list(maze_options.keys()) and user_input != 'OTHER':
    # Print the maze sizes in a visually appealing format
    print("Please choose a maze size from the options below:")
    print("  ┌────────────┬────────────┐")
    print("  │   Option   │   Size     │")
    print("  ├────────────┼────────────┤")
    for option, size in maze_options.items():
        print(f"  │     {option:<7}│   {size:<7}  │")
    print("  └────────────┴────────────┘")
    print("Or type 'other' to enter a size not listed")
    user_input = (input()).upper()
    
if user_input == 'OTHER':
    print("Please enter an edge size for the maze. \nNote that sizes over 100 are not recomended.")
    while not user_input.isdigit():
        user_input = input()

maze_sizes = {'S': (5,5),'M': (25,25),'L': (50,50)}
maze_speeds = {'S': 250 ,'M': 50,'L': 25}
maze_speed = int(maze_speed_eq(maze_sizes[user_input][0]))

# Set maze parameters
size = maze_sizes[user_input]
goal = (1,1)
start = (size[0],size[1])

# Create maze for the search algorithms
maze_search=maze(size[0],size[1])
maze_search.CreateMaze(goal[0],goal[1],loopPercent=50,theme="dark", saveMaze=True)
maze_file = find_maze_file()

textTitle(maze_search, "startup title", "")

# Solve the maze with each algorithm
run_algorithm(maze_search, start, goal, BFS)
#run_algorithm(maze_search, start, goal, DFS)
#run_algorithm(maze_search, start, goal, A_star)
#run_algorithm(maze_search, start, goal, value_iteration)
run_algorithm(maze_search, start, goal, policy_iteration)

maze_search.run()

#show_cell_values(maze_search, value_iteration_values)
show_policy(maze_search, policy_iteration_values)

delete_all_maze_files()
    
   