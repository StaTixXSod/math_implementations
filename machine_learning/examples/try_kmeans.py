from machine_learning.clustering.k_means import KMeans as kmeansmy
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from time import time

center_1 = np.array([1, 1])
center_2 = np.array([5, 5])
center_3 = np.array([8, 1])

data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200, 2) + center_2
data_3 = np.random.randn(200, 2) + center_3

df = np.concatenate((data_1, data_2, data_3), axis=0)

start = time()
print("[INFO] My KMEANS")
mymodel = kmeansmy(data=df, n_clusters=3)
mymodel.fit()
print(mymodel.cluster_centers)
print(mymodel.inertia)
print("[INFO] Time spent:", time() - start)

print("\n")
print("[INFO] Sklearn KMEANS")
skmodel = KMeans(n_clusters=3)
skmodel.fit(df)
print(skmodel.cluster_centers_)
print(skmodel.inertia_)

"""
[INFO] My KMEANS
[[5.04600269 4.91269894]
 [8.05816874 0.921641  ]
 [0.99888886 1.02990041]]
1130.0726820064901


[INFO] Sklearn KMEANS
[[5.04600269 4.91269894]
 [0.99888886 1.02990041]
 [8.05816874 0.921641  ]]
1130.0726820064904
"""