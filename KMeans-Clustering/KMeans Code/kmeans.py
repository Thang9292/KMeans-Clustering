import numpy as np

class KMeans(object):

    def __init__(self):
        pass

    def _init_centers(self, points, K, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments
        Return:
            centers: K x D numpy array, the centers.
        """
        N = points.shape[0]
        indices = np.random.randint(0, high=N, size=K)
        centers = points[indices]
        return centers

    def _kmpp_init(self, points, K, **kwargs):
        """
        k-means++ selects initial cluster centroids using sampling based on an empirical probability distribution of 
        the points contribution to the overall inertia. This technique speeds up convergence.
        Initialize one point at random. Loop for k - 1 iterations:
        Next, calculate for each point the distance of the point from its nearest center. Sample a point with a 
        probability proportional to the square of the distance of the point from its nearest center.

        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """

        N, D = points.shape
        centers = np.zeros((K, D))

        init_index = np.random.randint(0, high=N, size=1)
        centers[0] = points[init_index][np.newaxis, :] 

        for i in range(1, K):
            sq_dist = pairwise_dist(centers[i - 1].reshape(1, D), points)**2

            # Calculate the probabilities for each point to be chosen as the next center
            probs = sq_dist / np.sum(sq_dist)

            next_center_index = np.random.choice(N, p=probs[0])
            centers[i] = points[next_center_index]

        return centers

    def _update_assignment(self, centers, points):
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point
        """
        dist2 = pairwise_dist(points, centers)
        cluster_idx = np.argmin(dist2, axis = 1)
        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        """
        K, D = old_centers.shape
        N = points.shape[0]
        new_centers = np.zeros((K, D))
        counts = np.zeros(K)

        # Faster Implementation to count how many points are assigned to each new center
        # see https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
        counts = np.bincount(cluster_idx, minlength=K)

        # Mask to handle clusters with zero points
        mask = (counts != 0)
        
        # Faster Implementation to sum the points in respect to their center
        np.add.at(new_centers, cluster_idx, points)
        new_centers[mask] /= counts[mask, np.newaxis]

        # Re-initialize centers for empty clusters
        empty_clusters = np.where(counts == 0)[0]
        for cluster in empty_clusters:
            new_centers[cluster] = self._init_centers(points, 1)

        return new_centers

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """

        N, D = points.shape
        K = centers.shape[0]
        loss = 0

        for i in range(K):
            center = centers[i]
            cluster_points = points[cluster_idx == i]
            loss += np.sum(np.square(pairwise_dist(center[np.newaxis], cluster_points)))

        return loss


    def _get_centers_mapping(self, points, cluster_idx, centers):
        # create dict mapping each cluster to index to numpy array of points in the cluster
        centers_mapping = {key : [] for key in [i for i in range(centers.shape[0])]}
        for (p, i) in zip(points, cluster_idx):
            centers_mapping[i].append(p)
        for center_idx in centers_mapping:
            centers_mapping[center_idx] = np.array(centers_mapping[center_idx])
        self.centers_mapping = centers_mapping
        return centers_mapping

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, center_mapping=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        if center_mapping:
            return cluster_idx, centers, loss, self._get_centers_mapping(points, cluster_idx, centers)
        return cluster_idx, centers, loss


def pairwise_dist(x, y):
    np.random.seed(1)
    """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where pairwise_dist[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
    """
    n = x.shape[0]
    m = y.shape[0]

    # Reshaping x and y for broadcasting
    x_new = x[:, np.newaxis, :]  # Shape: (N, 1, D)
    y_new = y[np.newaxis, :, :]  # Shape: (1, M, D)

    dist = x_new - y_new                    # Shape: (N, M, D)
    dist_sq = dist ** 2                     # Shape: (N, M, D)
    sum_dist_sq = np.sum(dist_sq, axis = 2) # Shape: (N, M)
    pairwise_dist = np.sqrt(sum_dist_sq)

    return pairwise_dist


def silhouette_coefficient(points, cluster_idx, centers, centers_mapping):
    """
    Args:
        points: N x D numpy array
        cluster_idx: N x 1 numpy array
        centers: K x D numpy array, the centers
        centers_mapping: dict with K keys (cluster indicies) each mapping to a C_i x D 
        numpy array with C_i corresponding to the number of points in cluster i
    Return:
        silhouette_coefficient: final coefficient value as a float 
        mu_ins: N x 1 numpy array of mu_ins (one mu_in for each data point)
        mu_outs: N x 1 numpy array of mu_outs (one mu_out for each data point)
    """

    #Initialize all the variables
    K = centers.shape[0]
    N = points.shape[0]

    mu_ins = np.zeros((N, 1))
    mu_outs = np.ones((N, 1))

    
    mu_outTemp = 1
    silo_vals = np.zeros(N)

    for j in range(N):
        for i in range(K):
            if i == cluster_idx[j]:
                if centers_mapping[cluster_idx[j]].shape[0] == 1:
                    mu_ins[j] = np.sum(pairwise_dist(points[j][np.newaxis], centers_mapping[cluster_idx[j]]))
                else :
                    mu_inTemp = np.sum(pairwise_dist(points[j][np.newaxis], centers_mapping[cluster_idx[j]])) / (centers_mapping[cluster_idx[j]].shape[0] - 1)
                    mu_ins[j] = mu_inTemp
            else :
                mu_outTemp = np.mean(pairwise_dist(points[j][np.newaxis], centers_mapping[i]))
                if mu_outs[j] == 1:
                    mu_outs[j] = mu_outTemp

                if mu_outTemp < mu_outs[j]:
                     mu_outs[j] = mu_outTemp
        
        #Calculating S_i
        if (mu_ins[j] < mu_outs[j]):
            silo_vals[j] = (mu_outs[j] - mu_ins[j]) / mu_outs[j]
        else :
            silo_vals[j] = (mu_outs[j] - mu_ins[j]) / mu_ins[j]

    return np.sum(silo_vals) / N, mu_ins, mu_outs

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
