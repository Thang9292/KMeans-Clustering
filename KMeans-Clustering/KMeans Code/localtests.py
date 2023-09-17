import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from kmeans import *

def print_success_message():
    print("UnitTest passed successfully!")

class KMeansTests(unittest.TestCase):

    def runTest(self):
        pass

    def test_kmeans_loss(self, km=KMeans):
        points = np.array([[-1.43120297,  0.8786823],
                            [-0.92082436,  0.48852312],
                            [-0.98384412, -1.95809560],
                            [ 0.16539071, -2.00230059],
                            [ 1.13874072, -1.37183974],
                            [-0.30292898, -1.90092874],
                            [-1.96110084,  0.25292960],
                            [ 0.80891560, -0.43945057],
                            [-0.35643708, -0.16907999],
                            [-1.15222464,  1.23224667]])
        cluster_idx = np.array([2, 0, 2, 2, 0, 1, 2, 2, 2, 0])
        centers = np.array([[-0.29934073,  0.22471242],
                            [-0.65337045,  0.31246784],
                            [-0.24212177, -0.06978610]])
        loss = KMeans._get_loss(self, centers, cluster_idx, points)
        self.assertTrue(np.isclose(loss, 26.490707917359302), msg="Expected: 26.490707917359302 got: %s"
                                                                  % loss)
        print_success_message()

    def test_update_centers(self, km=KMeans):
        points = np.array([[-1.43120297,  0.8786823],
                            [-0.92082436,  0.48852312],
                            [-0.98384412, -1.95809560],
                            [ 0.16539071, -2.00230059],
                            [ 1.13874072, -1.37183974],
                            [-0.30292898, -1.90092874],
                            [-1.96110084,  0.25292960],
                            [ 0.80891560, -0.43945057],
                            [-0.35643708, -0.16907999],
                            [-1.15222464,  1.23224667]])
        cluster_idx = np.array([2, 0, 2, 2, 0, 1, 2, 2, 2, 0])
        old_centers = np.array([[-0.29934073,  0.22471242],
                                [-0.65337045,  0.31246784],
                                [-0.24212177, -0.06978610]])
        new_centers = KMeans._update_centers(self, old_centers, cluster_idx, points)
        expected_centers = [[-0.31143609,  0.11631002],
                             [-0.30292898, -1.90092874],
                             [-0.62637978, -0.57288581]]
        self.assertTrue(np.allclose(new_centers, expected_centers, atol=1e-4),
                        msg="Incorrect centers, check that means are computed correctly")
        print_success_message()

    def test_init(self, km=KMeans):
        X, y_true = make_blobs(n_samples=4000, centers=4, cluster_std=0.70, random_state=0)
        X = X[:, ::-1]
        # Plot init centers along side sample data

        plt.figure(1)
        colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

        for k, col in enumerate(colors):
            cluster_data = y_true == k
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

        # Calculate centers from randomly init centers
        centers_init1 = KMeans._init_centers(self, X, 4)
        try:
            # Calculate centers from kmpp init centers
            centers_init2 = KMeans._kmpp_init(self, X, 4)
            plt.scatter(centers_init1[:, 0], centers_init1[:, 1], c="k", s=50)
            plt.scatter(centers_init2[:, 0], centers_init2[:, 1], c="r", s=50)
            plt.legend(['1','2','3','4','random','km++'])
        except NotImplementedError:
            plt.scatter(centers_init1[:, 0], centers_init1[:, 1], c="k", s=50)
            plt.legend(['1','2','3','4','random'])
        finally:
            plt.title("K-Means++ Initialization")
            plt.xticks([])
            plt.yticks([])
            plt.show()
