from pyamaze import maze, agent, COLOR

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

# Solve the maze with BFS
def BFS(maze_map, start, goal):
    que = []
    checked_cells = []
    prev_pos = {}
    current_pos = start
    
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    
    # While the goal has not been reached & the que is not empty
    while((current_pos != goal) and (len(que) > 0)):
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        que = que + available_cells
        for cell in available_cells:
            prev_pos[cell] = current_pos
        
        # Get next cell from the que
        current_pos = que[0]
        del que[0]
        
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
    return directions

def DFS(maze_map, start, goal):
    que = []
    checked_cells = []
    prev_pos = {}
    current_pos = start
    
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    
    # While the goal has not been reached & the que is not empty
    while((current_pos != goal) and (len(que) > 0)):
        # Check neaby available cells
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        que = que + available_cells
        for cell in available_cells:
            prev_pos[cell] = current_pos
        
        # Get next cell from the que
        current_pos = que[len(que)-1]
        del que[len(que)-1]
        
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
    return directions

# Set variables
size = (8,8)
goal = (1,1)
start = (5,5)

# Create maze
m=maze(size[0],size[1])
m.CreateMaze(goal[0],goal[1],loopPercent=100,theme="dark")

# Create maze-solving agents
BFS_agent=agent(m,start[0],start[1],filled=True,footprints=True,color=COLOR.cyan)
DFS_agent=agent(m,start[0],start[1],filled=True,footprints=True,color=COLOR.red)

# Get the paths
BFS_path = BFS(m.maze_map, start, goal)
DFS_path = DFS(m.maze_map, start, goal)
print(BFS(m.maze_map, start, goal))

# Solve maze
m.tracePath({BFS_agent:BFS_path}, delay=100)
m.tracePath({DFS_agent:DFS_path}, delay=100)
m.run()
    
    