import numpy as np
import matplotlib
import pandas as pd
from tkinter import filedialog, simpledialog, Tk
import matplotlib.pyplot as plt
from datetime import datetime

matplotlib.rc('figure', figsize=(10, 5))
Tk().withdraw()
FILE_PATH = filedialog.askopenfilename()
fileValues = pd.read_csv(FILE_PATH, delimiter='\s+', header=None, names=["x", "y"])
num_cluster = simpledialog.askinteger("Input", "Write number of clusters!")
x = fileValues.iloc[:, [0, 1]].values
N = len(x)
y = np.zeros(N)
sse = []

def k_avg(num_cluster, x, y, N):
    flag = True
    isFirstIteration = True
    prevCentroid = None
    centroid = None
    avgArr = []
    avg = 0

    while flag:
        if isFirstIteration:
            isFirstIteration = False
            startPoint = np.random.choice(range(N), num_cluster, replace=False)
            centroid = x[startPoint]
        else:
            prevCentroid = np.copy(centroid)
            for i in range(num_cluster):
                centroid[i] = np.mean(x[y == i], axis=0)
        for i in range(N):
            dist = np.sum((centroid - x[i]) ** 2, axis=1)
            avgArr.append(min(dist))
            minInd = np.argmin(dist)
            y[i] = minInd
        if np.array_equiv(centroid, prevCentroid):
            avg = np.mean(avgArr)
            flag = False
    avgArr.clear()
    return avg, y, x


avg, y, x = k_avg(num_cluster, x, y, N)
now = datetime.now()
sse.append(avg)
for k in range(num_cluster):
    fig = plt.scatter(x[y == k, 0], x[y == k, 1])
plt.show()

for i in range(1, num_cluster):
    y = np.copy(y)
    avg, y, x = k_avg(i, x, y, N)
    sse.append(avg)

sse.sort(reverse=True)
print(list(range(1, num_cluster + 1)), sse)
plt.plot(list(range(1, num_cluster + 1)), sse)
plt.xticks(list(range(1, num_cluster + 1)))
plt.scatter(list(range(1, num_cluster + 1)), sse)
plt.xlabel("Number of clusters k")
plt.ylabel("Total Within Sum of square")
plt.show()

