from pyamaze import maze, agent, COLOR
from varname.helpers import Wrapper
import math

def solve_maze(maze, start, goal, search_func):
    # Define maze-solving agents
    search_agent = agent(maze,start[0],start[1],filled=True,footprints=True,color=COLOR.cyan)
    solve_agent = agent(maze,start[0],start[1],filled=True,footprints=True,color=COLOR.green)
    
    # Define the paths
    path, search = search_func(maze.maze_map, start, goal)
    search_func = Wrapper(dict())
    print(f"{search_func.name} path length: {len(path)}")
    
    # Display maze search and solve
    maze.tracePath({search_agent:search}, delay=1, kill=True)
    maze.tracePath({solve_agent:path}, delay=30, kill=True)

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
    return directions, past_cells

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
    return directions, past_cells

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

def get_lowest_cost_cell(cell_costs):
    # Use the min function with a key argument to get the key associated with the lowest value
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
    return directions, past_cells

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

def value_iteration(maze_map, start, goal):
    # Set variables
    num_iterations = 1
    discount = .9
    R = {}
    V = {}
    que = []
    checked_cells = []
    
    # Initialise variables
    current_pos = goal
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    R[goal] = 1
    
    for i in range(num_iterations):
        
        # Iterate over all cells
        while(len(checked_cells) < len(maze_map)-1):
            checked_cells.append(current_pos)
            available_cells = get_available_cells(maze_map, current_pos, checked_cells)
            neighbouring_cells = get_neighbouring_cells(maze_map, current_pos)
            que = que + available_cells
            
            V[current_pos] = bellman_eq(current_pos, neighbouring_cells, R, V, discount)
            print(current_pos, V[current_pos])
            
            # Get next unchecked cell from the que
            while que[0] in checked_cells:
                del que[0]
            current_pos = que[0]
    print(V)

# Set variables
size = (30,30)
goal = (1,1)
start = (30,30)

# Create maze
m=maze(size[0],size[1])
m.CreateMaze(goal[0],goal[1],loopPercent=30,theme="dark")

# Solve the maze with each algorithm
solve_maze(m, start, goal, BFS)
solve_maze(m, start, goal, DFS)
solve_maze(m, start, goal, A_star)

value_iteration(m.maze_map, start, goal)

#m.run()
    
    