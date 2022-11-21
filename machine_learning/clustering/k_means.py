from typing import Union
from machine_learning.distance_metrics import euclidean_distance
import copy
import numpy as np
import pandas as pd


class KMeans:
    def __init__(self,
                 data: Union[pd.DataFrame, np.ndarray],
                 n_clusters: int,
                 n_init: int = 10,
                 tolerance: float = 1e-3):
        self.data = data
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.tol = tolerance

        # Check if data is pd.DataFrame or np.array
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.to_numpy()
        assert isinstance(self.data, np.ndarray)

        # Initialize parameters
        self.cluster_centers = self._initialize_clusters()
        self.previous_cluster_centers = copy.deepcopy(self.cluster_centers)
        self.inertia = np.inf

    def fit(self):
        for i in range(self.n_init):
            cluster_centers, inertia, cluster_indices = self.fit_iter()
            if inertia < self.inertia:
                self.inertia = inertia
                self.cluster_centers = cluster_centers

        return self

    def fit_iter(self, max_iters: int = 100):
        """
        Fit the model (adjust cluster centers)

        Steps:
        ------
        1. Initialize the cluster centers with random values (did it in init method)
        2. Calculate the distance between data and cluster centers
        3. For each point assign the closest cluster
        4. Calculate centroids for each data by clusters
        5. Move cluster center to their centroids
        6. Continue the previous steps until the centers stops moving
        """

        # Initialize cluster centers
        cluster_center = self._initialize_clusters()
        prev_cluster_center = copy.deepcopy(cluster_center)

        # Restrict the loop to avoid infinite process
        for i in range(max_iters):
            distances = self._calculate_euclidian_distance(self.data, cluster_center)
            assigned_classes = self._assign_cluster_to_data(distances)
            centroids = self._calculate_centroids(assigned_classes)

            # If any value in centroid is nan -> reset
            if np.isnan(centroids).any():
                cluster_center = self._initialize_clusters()
                prev_cluster_center = copy.deepcopy(cluster_center)
                continue

            # Move cluster centers to their centroids
            cluster_center = copy.deepcopy(centroids)

            # Check if the centroids have moved
            if self._get_distance_diff(cluster_center, prev_cluster_center) < self.tol:
                # If centers doesn't change, calculate inertia and return results
                inertia = self._get_inertia(centroids, assigned_classes)
                return centroids, inertia, assigned_classes

            prev_cluster_center = copy.deepcopy(cluster_center)
            if i == (max_iters - 1):
                print("[INFO] The process has reached the end of loop and didn't converge")

        inertia = self._get_inertia(centroids, assigned_classes)
        return centroids, inertia, assigned_classes

    def _get_inertia(self, centroids, assigned_classes) -> float:
        """
        Inertia is the metric that show how far our data from their clusters.

        Steps:
        ------
        To calculate inertia:\n
        1. Calculate Euclidian distance between data and centroid\n
        2. Square this distance\n
        3. Sum it all together\n

        The lower the inertia -> the better the model.
        """
        distances_by_clusters = []
        for i in range(self.n_clusters):
            cluster_data = self.data[assigned_classes == i]
            centroid = centroids[i]
            distance = euclidean_distance(cluster_data, centroid)
            squared_distance = np.power(distance, 2)
            distances_by_clusters.append(squared_distance)

        return np.array(distances_by_clusters).sum()

    def _initialize_clusters(self) -> np.ndarray:
        """
        Initialize array with random values
        Return the array with shape (rows - number of clusters,
                                     columns - number of items)

        """
        dim = self.data.shape[1]  # number of items in row
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        cluster_centers = np.random.randn(self.n_clusters, dim) * std + mean
        return cluster_centers

    def _initialize_distance_matrix(self):
        """
        Return the array, that will contain the distance for each cluster
        between data point and cluster point. Shape: (number of rows, number of clusters).
        This means that for each row we'll write the distance to each cluster
        """
        n_rows = self.data.shape[0]
        return np.zeros(shape=(n_rows, self.n_clusters))

    def _calculate_euclidian_distance(self, data, clusters, fast_method: bool = True) -> np.ndarray:
        """
        Here we calculate the distance between each data point (the row) and
        the cluster point. Each value are written in the `distance` variable.
        """
        distances = self._initialize_distance_matrix()

        if fast_method:
            for i in range(self.n_clusters):
                distances[:, i] = np.linalg.norm(data - clusters[i], axis=1)

        else:  # The slowest method, but have more description in code
            for row_idx in range(self.data.shape[0]):
                for cluster_idx in range(self.n_clusters):
                    data_point = data[row_idx]
                    cluster_point = clusters[cluster_idx]
                    distances[row_idx, cluster_idx] = euclidean_distance(data_point, cluster_point)

        return distances

    def _assign_cluster_to_data(self, distances: np.ndarray):
        """For each row returns the index of cluster, that have the minimal distance"""
        assigned_classes = np.argmin(distances, axis=1)
        return assigned_classes

    def _calculate_centroids(self, assigned_classes):
        """Return the center of mass for each cluster"""
        centroids = np.zeros_like(self.cluster_centers)
        for i in range(self.n_clusters):
            cluster_data = self.data[assigned_classes == i]
            centroids[i] = np.mean(cluster_data, axis=0)
        return centroids

    def _move_clusters_to_centroids(self, centroids) -> None:
        """Set cluster centers to centroid points"""
        self.cluster_centers = centroids
        return

    def _get_distance_diff(self, current, previous):
        """Return the difference between the previous centers and current centers"""
        return np.linalg.norm(current - previous)
