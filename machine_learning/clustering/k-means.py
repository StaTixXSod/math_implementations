import copy
from typing import Union, Tuple
from machine_learning.distance_metrics import euclidean_distance
import numpy as np
import pandas as pd

np.random.seed(42)


class KMeans:
    def __init__(self, n_clusters: int, tolerance: float = 1e-3):
        self.n_clusters = n_clusters
        self.tol = tolerance

    def fit(self, x_train: Union[pd.DataFrame, np.ndarray]):
        """
        Fit the model (adjust cluster centers)

        Steps:
        ------
        1. Initialize the cluster centers with random values
        2. For each point calculate the euclidean distance to every cluster center
        3. For each point assign the closest cluster
        4. Calculate centroids for each data by clusters
        5. Move cluster center to their centroids
        6. Continue the previous steps until the centers stops moving
        """
        # Check if the data is np array or pd dataframe
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()
        assert isinstance(x_train, np.ndarray)

        # Start calculations
        cluster_centers, previous_cluster_centers = self._initialize_clusters(x_train)
        distances = self._initialize_distance_matrix(x_train)

        while True:
            distances = self._calculate_euclidian_distance(x_train, cluster_centers, distances)
            cluster_indices = self._assign_clusters(distances)
            centroids = self._calculate_centroids(x_train, cluster_indices)
            cluster_centers = self._move_clusters_to_their_centroids(cluster_centers, centroids)

            if self._get_distance_diff(cluster_centers, previous_cluster_centers) < self.tol:
                break
            previous_cluster_centers = copy.deepcopy(cluster_centers)

        return cluster_indices

    def predict(self, X_test):
        pass

    def _initialize_clusters(self, x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize array with random values
        Return the array with shape (rows - number of clusters,
                                     columns - number of items)

        """
        dim = x_train.shape[1]  # number of items in row
        min_item = np.amin(x_train)
        max_item = np.amax(x_train)
        cluster_centers = np.random.randint(low=min_item,
                                            high=max_item,
                                            size=(self.n_clusters, dim))
        return cluster_centers, cluster_centers

    def _initialize_distance_matrix(self, x_train):
        """
        Return the array, that will contain the distance for each cluster
        between data point and cluster point. Shape: (number of rows, number of clusters).
        This means that for each row we'll write the distance to each cluster
        """
        n_rows = x_train.shape[0]
        return np.zeros(shape=(n_rows, self.n_clusters))

    def _calculate_euclidian_distance(self,
                                      x_train: np.ndarray,
                                      cluster_centers: np.ndarray,
                                      distances: np.ndarray) -> np.ndarray:
        """
        Here we calculate the distance between each data point (the row) and
        the cluster point. Each value are written in the `distance` variable.
        """
        for row_idx in range(x_train.shape[0]):
            for cluster_idx in range(self.n_clusters):
                data_point = x_train[row_idx]
                cluster_point = cluster_centers[cluster_idx]
                point_distance = euclidean_distance(data_point, cluster_point)
                distances[row_idx, cluster_idx] = point_distance

        return distances

    def _assign_clusters(self, distances: np.ndarray):
        """For each row returns the index of cluster, that have the minimal distance"""
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, x_train: np.ndarray, cluster_indices: np.ndarray):
        """Return the center of mass for each cluster"""
        dim = x_train.shape[0]
        centroid = np.zeros(shape=(self.n_clusters, dim))
        for i in range(self.n_clusters):
            cluster_data = x_train[cluster_indices == i]
            mean_data = np.mean(cluster_data, axis=0)
            centroid[i] = mean_data

        return centroid

    def _move_clusters_to_their_centroids(self, cluster_centers, centroids):
        """Return the cluster array with new centers"""
        return centroids

    def _get_distance_diff(self, current_centroids, previous_centroids):
        """Return the difference between the previous centers and current centers"""
        diff = current_centroids - previous_centroids
        summed_diff = np.sum(diff)
        return abs(summed_diff)


df = pd.DataFrame(
    data=[
        [0, 3, 4, 3, 1],
        [3, 0, 1, 2, 5],
        [4, 1, 0, 3, 3],
        [3, 2, 3, 0, 4],
        [1, 5, 3, 4, 0]
    ],
    columns=['A', 'B', 'C', 'D', 'E'],
    index=['A', 'B', 'C', 'D', 'E']
)

print(df)

model = KMeans(3)
print(model.fit(df))
