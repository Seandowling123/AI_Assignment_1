from pyamaze import maze, agent, COLOR

m=maze(5,5)

goal = (1,1)
start = (5,5)

m.CreateMaze(goal[0],goal[1],theme="light")
a=agent(m,start[0],start[1],filled=True,footprints=True,color=COLOR.red)
b=agent(m,filled=True,footprints=True,color=COLOR.cyan)

path = 'WWNWNNENW'
path2 = 'WNWNWNNENW'
m.tracePath({a:path}) 

#m.run()

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

def BFS(maze_map, start, goal):
    que = []
    checked_cells = []
    prev_pos = {}
    current_pos = start
    que = que + get_available_cells(maze_map, current_pos, checked_cells)
    print(que)
    
    while((current_pos != goal) and (len(que) > 0)):
        print(current_pos)
        checked_cells.append(current_pos)
        available_cells = get_available_cells(maze_map, current_pos, checked_cells)
        que = que + available_cells
        for cell in available_cells:
            prev_pos[cell] = current_pos
        
        current_pos = que[0]
            
        del que[0]
    
    if current_pos == goal:    
        path = []
        dummy = current_pos
        path.append(dummy)
        while dummy != start:
            prev = prev_pos[dummy]
            path.append(prev)
            dummy = prev
        
    return path[::-1]

print(BFS(m.maze_map, start, goal))
    
    