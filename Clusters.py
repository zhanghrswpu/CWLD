from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
def cls(data, cluster, n_clusters):
    if cluster == 'Birch':
        estimator = Birch(n_clusters=n_clusters)
        estimator.fit(data)
        return estimator.labels_
    elif cluster == 'MiniBatchKMeans':
        random_state = 2187
        estimator = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
        estimator.fit(data)
        return estimator.labels_
    elif cluster == 'AgglomerativeClustering':
        estimator = AgglomerativeClustering(n_clusters)
        estimator.fit(data)
        return estimator.labels_
    elif cluster == 'SpectralClustering':
        estimator = SpectralClustering(n_clusters)
        estimator.fit(data)
        return estimator.labels_
    elif cluster == 'DBSCAN':
        estimator = DBSCAN(eps=0.0001, min_samples = 20)
        estimator.fit(data)
        return estimator.labels_
    else:
        random_state = 2187
        estimator = KMeans(n_clusters, random_state=random_state)
        estimator.fit(data)
        return estimator.labels_