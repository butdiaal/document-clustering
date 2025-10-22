from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from config import *
from sentence_transformers import SentenceTransformer

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


class HierarchicalClusteringStrategy(ClusteringStrategy):
    """Иерархическая кластеризация"""

    def __init__(self, n_clusters=None, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def cluster(self, features):
        if hasattr(features, 'toarray'):
            features = features.toarray()

        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(features) // 3))

        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.linkage
        )
        cluster_labels = hierarchical.fit_predict(features)
        return cluster_labels


class SemanticClusteringStrategy(ClusteringStrategy):
    """Семантическая кластеризация для создания мета-абзацев"""

    def __init__(self, eps=0.5, min_samples=2, model_name=BERT_MODEL_NAME):
        self.eps = eps
        self.min_samples = min_samples
        self.model = SentenceTransformer(model_name)

    def cluster(self, features):
        """Кластеризует текстовые фрагменты по семантической близости"""

        if len(features) <= 1:
            return [0] * len(features) if features else []

        if hasattr(features, 'shape'):
            embeddings = features
        else:
            embeddings = self.model.encode(features, normalize_embeddings=True)

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(embeddings)

        cluster_labels = [label if label != -1 else max(cluster_labels) + 1 for label in cluster_labels]

        return cluster_labels