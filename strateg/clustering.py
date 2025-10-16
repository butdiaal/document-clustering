from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

class ClusteringStrategy(ABC):
    """Абстрактный класс стратегии кластеризации"""

    @abstractmethod
    def cluster(self, features):
        """Кластеризует тексты и возвращает метки кластеров"""
        pass


class KMeansClusteringStrategy(ClusteringStrategy):
    """Стратегия кластеризации K-Means"""

    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters

    def cluster(self, features):
        if hasattr(features, 'toarray'):
            features = features.toarray()

        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(features) // 3))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        return cluster_labels


class HierarchicalClusteringStrategy(ClusteringStrategy):
    """Иерархическая кластеризация"""
    pass


class DBSCANClusteringStrategy(ClusteringStrategy):
    """Плотностная кластеризация DBSCAN"""

    def __init__(self, eps=0.5, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, features):
        if hasattr(features, 'toarray'):
            features = features.toarray()

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(features)

        cluster_labels = [label if label != -1 else max(cluster_labels) + 1 for label in cluster_labels]
        return cluster_labels