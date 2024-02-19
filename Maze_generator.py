from pyamaze import maze, agent, COLOR, textLabel, textTitle
import random
import time
import math
import os

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
        
# Display the maze being solved
def solve_maze(maze, start, goal, search_func):
    # Define the paths
    algo_name, path, search = search_func(maze.maze_map, start, goal)
    print(f"{algo_name} path length: {len(path)}")
    
    # Define maze-solving agents
    search_agent = agent(maze,start[0],start[1],filled=True,footprints=True,color=COLOR.cyan,name=algo_name)
    solve_agent = agent(maze,start[0],start[1],filled=True,footprints=True,color=COLOR.green,name=algo_name)
    
    # Display maze search and solve
    if algo_name != "Value Iteration":
        maze.tracePath({search_agent:search}, delay=1, kill=True)
        maze.tracePath({solve_agent:path}, delay=15, kill=True)
    else:
        maze.tracePath({search_agent:path}, delay=15, kill=True)
    
    
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

# Solve the maze with BFS
def BFS(maze_map, start, goal):
    que = []
    checked_cells = []
    prev_pos = {}
    
    # Initialise variables
    current_pos = start
    past_cells = [current_pos]
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    
    # While the goal has not been reached & the que is not empty
    while((current_pos != goal) and (len(que) > 0)):
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        que = que + available_cells
        for cell in available_cells:
            prev_pos[cell] = current_pos
        
        # Get next unchecked cell from the que
        while que[0] in checked_cells:
            del que[0]
        current_pos = que[0]
        past_cells.append(current_pos)
        
        if current_pos in checked_cells:
            print("fuck")
        
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
def DFS(maze_map, start, goal):
    que = []
    checked_cells = []
    prev_pos = {}
    
    # Initialise variables
    current_pos = start
    past_cells = [current_pos]
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    
    # While the goal has not been reached & the que is not empty
    while((current_pos != goal) and (len(que) > 0)):
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        que = que + available_cells
        for cell in available_cells:
            prev_pos[cell] = current_pos
        
        # Get next unchecked cell from the que
        while que[len(que)-1] in checked_cells:
            del que[len(que)-1]
        current_pos = que[len(que)-1]
        del que[len(que)-1]
        past_cells.append(current_pos)
        
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

# Calculates euclidean distance between a cell and the goal
def get_heuristic(cell, goal):
    distance = math.sqrt((cell[0] - goal[0])**2 + (cell[1] - goal[1])**2)
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
def A_star(maze_map, start, goal):
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
    
    # While the goal has not been reached & the que is not empty
    while(current_pos != goal):
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        
        # Calculate heuristics
        g_values.update(get_g(available_cells, current_pos, g_values))
        cell_costs.update(get_cell_costs(available_cells, g_values, cell_costs, goal))
        
        # Set previous postions
        for cell in available_cells:
            prev_pos[cell] = current_pos
    
        # Get next unchecked cell from the que
        while get_lowest_cost_cell(cell_costs) in checked_cells:
            del cell_costs[get_lowest_cost_cell(cell_costs)]
        current_pos = get_lowest_cost_cell(cell_costs)
        del cell_costs[current_pos]
        past_cells.append(current_pos)
        
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

# Creates a new maze
def create_maze(starter_maze):
    maze_file = find_maze_file()
    new_maze=maze(starter_maze.rows,starter_maze.cols)
    new_maze.CreateMaze(goal[0],goal[1],loopPercent=30,theme="dark", loadMaze=maze_file)
    return new_maze

# Display the value iteration values on the maze
def show_cell_values(maze, values):
    maze = create_maze(maze)
    
    cell_agents = []
    for value in values:
        cell_agent = agent(maze,value[0],value[1],filled=True,footprints=False,color=COLOR.from_value(values[value]))
        cell_agents.append(cell_agent)
        
    for cell_agent in cell_agents:
        maze.tracePath({cell_agent:[(0,0)]}, delay=1, kill=False)
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

# Get the solve the maze from the cell values
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
        
# Calculate the value for a cell
def bellman_eq(current_pos, neighbouring_cells, R, V, discount):
    Q_values = []
    if current_pos in R:
        cell_R = R[current_pos]
    else: cell_R = 0
    
    # Calculate Bellman eq
    for cell in neighbouring_cells:
        cell_V = 0
        if cell in V:
            cell_V  = V[cell]
        Q_values.append(cell_R+discount*cell_V)
    return max(Q_values)

# Solve the maze with value iteration
def value_iteration(maze_map, start, goal):
    print(maze_map)
    discount = .9
    R = {}
    V = {}
    old_V = 0
    delta = 0
    deltas = []
    R[goal] = 1
    delta = 1
    epsilon = .0001
    iterations = 0
    
    # Calculate the values for each cell in the maze until convergence
    while delta > epsilon:
        iterations = iterations+1
        deltas = []
        old_V = V.copy()
        
        for cell in maze_map:
            neighbouring_cells = get_neighbouring_cells(maze_map, cell)
            V[cell] = bellman_eq(cell, neighbouring_cells, R, old_V, discount)
            
            # Calculate convergence
            if cell in old_V:
                cell_delta = V[cell] - old_V[cell]
            else: cell_delta = V[cell]
            deltas.append(cell_delta)
        delta = abs(sum(deltas)/len(V))
    
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
def policy_evaluation(cell, maze_map, policy, R, old_V, discount):
    V = 0
    if not maze_map[cell][policy[cell]]:
        old_V = {}
    V = bellman_eq(cell, get_cell_from_direction(cell,[policy[cell]]), R, old_V, discount)
    return V

# Find the optimal policy for each cell
def get_optimal_policy(cell, neighbouring_cells, V):
    highest_V_cell = 0
    for neigbouring_cell in neighbouring_cells:
        if V[neigbouring_cell] > V[highest_V_cell]:
            highest_V_cell = neigbouring_cell
    direction = to_directions([cell, highest_V_cell])
    return direction

# Update the policy of each cell to maximise value
def policy_improvement( maze_map, policy, R, V):
    new_policy = {}
    for cell in maze_map:
        neighbouring_cells = get_neighbouring_cells(maze_map, cell)
        new_policy[cell] = get_optimal_policy(cell, neighbouring_cells, V)
    return new_policy
        
        
# Solve the maze with policy iteration
def policy_iteration(maze_map, start, goal):
    discount = .9
    R = {}
    V = {}
    R[goal] = 1
    policy = initialise_policy(maze_map)
    iterations = 0
    
    # Update the policy until convergence]
    for i in range(1,3):
        iterations = iterations+1
        old_V = V.copy()
        
        for cell in maze_map:
            V[cell] = policy_evaluation(cell, maze_map, policy, R, old_V, discount)
        print(V)
    
    
# Set variables
size = (30,30)
goal = (1,1)
start = (30,30)

# Create maze for the search algorithms
maze_search=maze(size[0],size[1])
maze_search.CreateMaze(goal[0],goal[1],loopPercent=30,theme="dark", saveMaze=True)
maze_file = find_maze_file()
textLabel(maze_search, "title", "val")
textTitle(maze_search, "startup title", "")

# Solve the maze with each algorithm
#solve_maze(maze_search, start, goal, BFS)
#solve_maze(maze_search, start, goal, DFS)
#solve_maze(maze_search, start, goal, A_star)
#solve_maze(maze_search, start, goal, value_iteration)
maze_search.run()

policy_iteration(maze_search.maze_map, start, goal)

# Create maze for the MDP algorithms
#maze_MDP=maze(size[0],size[1])
#maze_MDP.CreateMaze(goal[0],goal[1],loopPercent=30,theme="dark", loadMaze=maze_file)

#textLabel(maze_MDP, "cock", "cock2")
#show_cell_values(maze_MDP, V)

#maze_MDP.run()

delete_all_maze_files()
    
   