from pyamaze import maze,COLOR,agent

m=maze(5,5)
m.CreateMaze(theme="light")
a=agent(m,filled=True,footprints=True,color=COLOR.red)
b=agent(m,filled=True,footprints=True,color=COLOR.cyan)

path = 'WWNWNNENW'
path2 = 'WNWNWNNENW'
m.tracePath({a:path}) 

print(m.path)
m.run()
print(m.maze_map)

# The plan: MAke 3 func BFS, DFS and A* return a string to solve the maze map