from sklearn.cluster import KMeans
from machine_learning.clustering.k_means import KMeans as kmeansmy
from time import time
import numpy as np


X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

start = time()
print("[INFO] My KMEANS")
my_kmeans = kmeansmy(data=X, n_clusters=2).fit()
print("[INFO] Labels:", my_kmeans.cluster_labels)
print("[INFO] Predict:", my_kmeans.predict([[0, 0], [12, 3]]))
print("[INFO] Centers:", my_kmeans.cluster_centers)
print("[INFO] Time spent:", time() - start)

print("\n")
start = time()
print("[INFO] Sklearn KMEANS")
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("[INFO] Labels:", kmeans.labels_)
print("[INFO] Predict:", kmeans.predict([[0, 0], [12, 3]]))
print("[INFO] Centers:", kmeans.cluster_centers_)
print("[INFO] Time spent:", time() - start)

"""
[INFO] My KMEANS
[INFO] Labels: [0 0 0 1 1 1]
[INFO] Predict: [0 1]
[INFO] Centers: [[ 1.  2.]
 [10.  2.]]
[INFO] Time spent: 0.005308866500854492


[INFO] Sklearn KMEANS
[INFO] Labels: [1 1 1 0 0 0]
[INFO] Predict: [1 0]
[INFO] Centers: [[10.  2.]
 [ 1.  2.]]
[INFO] Time spent: 0.10580325126647949
"""