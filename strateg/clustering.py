from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from config import *
import numpy as np
import hdbscan
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
        if hasattr(features, "toarray"):
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
        if hasattr(features, "toarray"):
            features = features.toarray()

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(features)

        cluster_labels = [
            label if label != -1 else max(cluster_labels) + 1
            for label in cluster_labels
        ]
        return cluster_labels


class HierarchicalClusteringStrategy(ClusteringStrategy):
    """Иерархическая кластеризация"""

    def __init__(self, n_clusters=None, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def cluster(self, features):
        if hasattr(features, "toarray"):
            features = features.toarray()

        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(features) // 3))

        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=self.linkage
        )
        cluster_labels = hierarchical.fit_predict(features)
        return cluster_labels


class SemanticClusteringStrategy(ClusteringStrategy):
    """Семантическая кластеризация для создания мета-абзацев"""

    def __init__(self, eps=None, min_samples=None, model_name=BERT_MODEL_NAME):
        self.eps = eps or EPS_DBSCAN
        self.min_samples = min_samples or MIN_SAMPLES_DBSCAN
        self.model = SentenceTransformer(model_name)

    def cluster(self, features):
        """Кластеризует текстовые фрагменты по семантической близости"""

        if len(features) <= 1:
            return [0] * len(features) if features else []

        if hasattr(features, "shape"):
            embeddings = features
        else:
            embeddings = self.model.encode(
                features, normalize_embeddings=True, show_progress_bar=False
            )

        best_labels = None

        for eps in [0.3, 0.4, 0.5, 0.6]:
            dbscan = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(embeddings)

            unique = np.unique(labels)
            n_clusters = len(unique) - (1 if -1 in unique else 0)

            if n_clusters >= 3:
                best_labels = labels
                break
            elif best_labels is None:
                best_labels = labels

        if best_labels is None:
            best_labels = labels

        cluster_labels = self._assign_noise_to_clusters(best_labels, embeddings)

        return cluster_labels

    def _assign_noise_to_clusters(self, cluster_labels, embeddings):
        """Присваивает шумовые точки ближайшим кластерам"""
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        noise_mask = cluster_labels == -1
        if not noise_mask.any():
            return cluster_labels

        cluster_mask = noise_mask
        if not cluster_mask.any():
            return np.zeros_like(cluster_labels)

        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(embeddings[cluster_mask])
        distances, indices = neighbors.kneighbors(embeddings[noise_mask])

        new_labels = cluster_labels.copy()
        cluster_indices = np.where(cluster_mask)[0]

        for i, noise_idx in enumerate(np.where(noise_mask)[0]):
            nearest_idx = cluster_indices[indices[i][0]]
            new_labels[noise_idx] = cluster_labels[nearest_idx]

        return new_labels
