import time

for i in range(100):
    x = i**6
    times = time.time()
    print(len(str(times)), times)
