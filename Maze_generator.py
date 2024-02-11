from pyamaze import maze,COLOR,agent

m=maze(5,5)
m.CreateMaze()
a=agent(m,footprints=True)

path = 'EENWWSES'

m.run()