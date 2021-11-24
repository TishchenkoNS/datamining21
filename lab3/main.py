import numpy as np
import matplotlib
import pandas as pd
from tkinter import filedialog, simpledialog, Tk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

matplotlib.rc('figure', figsize=(10, 5))
Tk().withdraw()
FILE_PATH = filedialog.askopenfilename()
fileValues = pd.read_csv(FILE_PATH, delimiter='\s+', header=None, names=["x", "y"])
clustersNumber = simpledialog.askinteger("Input", "Write number of clusters!")
x = fileValues.iloc[:, [0, 1]].values
N = len(x)
y = np.zeros(N)
twss = []


def k_avg(clustersNumber, x, y, N):
    flag = True
    isFirstIteration = True
    prevCentroid = None
    centroid = None
    avgArr = []
    avg = 0

    while flag:
        if isFirstIteration:
            isFirstIteration = False
            startPoint = np.random.choice(range(N), clustersNumber, replace=False)
            centroid = x[startPoint]
        else:
            prevCentroid = np.copy(centroid)
            for i in range(clustersNumber):
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

for k in range(clustersNumber):
    fig = plt.scatter(x[y == k, 0], x[y == k, 1])
plt.show()

avg, y, x = k_avg(clustersNumber, x, y, N)
twss.append(avg)
for k in range(clustersNumber):
    fig = plt.scatter(x[y == k, 0], x[y == k, 1])
plt.show()

for i in range(1, clustersNumber):
    y = np.copy(y)
    avg, y, x = k_avg(i, x, y, N)
    twss.append(avg)

twss.sort(reverse=True)
print(list(range(1, clustersNumber + 1)), twss)
plt.plot(list(range(1, clustersNumber + 1)), twss)
plt.xticks(list(range(1, clustersNumber + 1)))
plt.scatter(list(range(1, clustersNumber + 1)), twss)
plt.xlabel("Number of clusters k")
plt.ylabel("Total Within Sum of square")
plt.show()

# sse = []
# k = range(1, clustersNumber + 1)
# for i in k:
#     km = KMeans(
#         n_clusters=i, init="random",
#         n_init=10, max_iter=300,
#         tol=1e-04, random_state=0
#     )
#     km.fit(x)
#     sse.append(km.inertia_)
# plt.plot(k, sse, marker='o')
# plt.xlabel('Num of Clusters')
# plt.ylabel('SSE')
# plt.show()
