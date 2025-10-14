from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class ClusteringStrategy(ABC):
    """Абстрактный класс стратегии кластеризации"""

    @abstractmethod
    def cluster(self, texts):
        """Кластеризует тексты и возвращает метки кластеров"""
        pass


class KMeansClusteringStrategy(ClusteringStrategy):
    """Стратегия кластеризации K-Means с TF-IDF"""

    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters

    def cluster(self, texts):
        """Кластеризация K-Means"""
        if not texts:
            return []

        if len(texts) <= 1:
            return [0] * len(texts)

        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = min(8, max(2, len(texts) // 3))

        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.8,
            stop_words=None,
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        return cluster_labels


class FixedClusteringStrategy(ClusteringStrategy):
    """Стратегия с фиксированным количеством кластеров"""

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters

    def cluster(self, texts):
        """Кластеризация с фиксированным количеством кластеров"""
        if not texts:
            return []

        if len(texts) <= 1:
            return [0] * len(texts)

        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.8,
            stop_words=None,
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(texts)),
                        random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        return cluster_labels
