from abc import ABC, abstractmethod
from collections import defaultdict

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
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


#
# class KMeansClusteringStrategy(ClusteringStrategy):
#     """Стратегия кластеризации K-Means"""
#
#     def __init__(self, n_clusters=None):
#         self.n_clusters = n_clusters
#
#     def cluster(self, features):
#         if hasattr(features, "toarray"):
#             features = features.toarray()
#
#         n_clusters = self.n_clusters
#         if n_clusters is None:
#             n_clusters = min(8, max(2, len(features) // 3))
#
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         cluster_labels = kmeans.fit_predict(features)
#         return cluster_labels


class DBSCANClusteringStrategy(ClusteringStrategy):
    """Плотностная кластеризация DBSCAN"""

    def __init__(self, eps=None, min_samples=None):
        self.eps = eps or EPS_DBSCAN
        self.min_samples = min_samples or MIN_SAMPLES_DBSCAN

    def cluster(self, features):
        """Кластеризация предложений через DBSCAN"""
        if hasattr(features, "toarray"):
            features = features.toarray()

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(features)

        return cluster_labels


class HDBSCANClusteringStrategy(ClusteringStrategy):
    """Плотностная кластеризация HDBSCAN"""

    def __init__(self, min_cluster_size=None, min_samples=None):
        self.hdbscan = hdbscan
        self.min_cluster_size = min_cluster_size or MIN_CLUSTER_SIZE
        self.min_samples = min_samples or MIN_SAMPLES_DBSCAN

    def cluster(self, features):
        """Кластеризация предложений через HDBSCAN"""
        if hasattr(features, "toarray"):
            features = features.toarray()

        clusterer = self.hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, min_samples=self.min_samples
        )
        cluster_labels = clusterer.fit_predict(features)

        return cluster_labels


class HierarchicalClusteringStrategy(ClusteringStrategy):
    """Иерархическая агломеративная кластеризация"""

    def __init__(self, distance_threshold=None, linkage=None, normalize=True):
        self.distance_threshold = distance_threshold or HIERARCHICAL_DISTANCE_THRESHOLD
        self.linkage = linkage or HIERARCHICAL_LINKAGE
        self.normalize = normalize

    def cluster(self, features):
        """Иерархическая кластеризация с нормализацией"""
        if hasattr(features, "toarray"):
            features = features.toarray()

        if self.normalize:
            if features.std() > 1.0:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)

                if self.distance_threshold > 5.0:
                    self.distance_threshold = 1.0

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            metric="euclidean",
        )

        cluster_labels = clustering.fit_predict(features)
        print(cluster_labels)
        return cluster_labels


class SemanticClusteringStrategy(ClusteringStrategy):
    """Семантическая кластеризация для создания мета-абзацев через предложения"""

    def __init__(self, eps=None, min_samples=None, model_name=BERT_MODEL_NAME):
        self.eps = eps or EPS_COMBINED
        self.min_samples = min_samples or MIN_SAMPLES_COMBINED

        self.model = SentenceTransformer(model_name)

    def cluster(self, features):
        """Кластеризация предложений через DBSCAN для семантической группировки"""
        if len(features) <= 1:
            return [0] * len(features) if features else []

        if hasattr(features, "shape"):
            embeddings = features
        else:
            embeddings = self.model.encode(features, normalize_embeddings=True)

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(embeddings)

        return cluster_labels

    def cluster_paragraphs_sentences(self, original_paragraphs, original_sentences):
        """Группировка абзацев через кластеризацию предложений"""
        sentences_data = original_sentences
        original_paragraphs = original_paragraphs

        if len(sentences_data) < MIN_SENTENCES_PER_CLUSTER:
            return original_paragraphs

        sentences = [data["sentence"] for data in sentences_data]
        cluster_labels = self.cluster(sentences)

        cluster_to_paragraphs = {}

        for data, cluster_id in zip(sentences_data, cluster_labels):
            if cluster_id == -1:
                continue

            para_idx = data["para_idx"]
            if cluster_id not in cluster_to_paragraphs:
                cluster_to_paragraphs[cluster_id] = set()
            cluster_to_paragraphs[cluster_id].add(para_idx)

        valid_clusters = {}
        for cluster_id, paragraph_indices in cluster_to_paragraphs.items():
            if len(paragraph_indices) >= 2:
                valid_clusters[cluster_id] = paragraph_indices

        if not valid_clusters:
            return original_paragraphs

        used_paragraphs = set()
        final_paragraphs = []
        paragraph_mapping = {}

        for cluster_id, para_indices in valid_clusters.items():
            cluster_paragraphs = []
            used_in_this_cluster = set()
            for para_idx in sorted(para_indices):
                if para_idx not in used_paragraphs:
                    cluster_paragraphs.append(original_paragraphs[para_idx])
                    used_in_this_cluster.add(para_idx)
                    used_paragraphs.add(para_idx)

            if cluster_paragraphs:
                meta_paragraph = "\n\n".join(cluster_paragraphs)
                if len(meta_paragraph) >= MIN_LENGTH:
                    final_paragraphs.append(meta_paragraph)
                    paragraph_mapping[len(final_paragraphs) - 1] = used_in_this_cluster

        for para_idx, paragraph in enumerate(original_paragraphs):
            if para_idx not in used_paragraphs and len(paragraph) >= MIN_LENGTH:
                final_paragraphs.append(paragraph)

        return final_paragraphs, paragraph_mapping
