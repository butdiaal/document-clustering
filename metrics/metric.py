import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from config import BERT_MODEL_NAME
from collections import Counter


class ClusteringMetrics:
    """Класс для расчета метрик качества кластеризации"""

    def __init__(self, model_name=BERT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def calculate_internal_metrics(self, texts, cluster_labels, embeddings=None):
        """Внутренние метрики (не требуют эталонной разметки)"""
        if embeddings is None:
            embeddings = self.model.encode(texts, normalize_embeddings=True)

        unique_labels = set(cluster_labels)
        valid_indices = [
            i
            for i, label in enumerate(cluster_labels)
            if list(cluster_labels).count(label) > 1
        ]

        if len(valid_indices) < 2 or len(unique_labels) < 2:
            return {}

        valid_embeddings = embeddings[valid_indices]
        valid_labels = [cluster_labels[i] for i in valid_indices]

        metrics = {}

        try:
            metrics["silhouette_score"] = silhouette_score(
                valid_embeddings, valid_labels
            )
        except:
            metrics["silhouette_score"] = -1

        try:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                embeddings, cluster_labels
            )
        except:
            metrics["calinski_harabasz_score"] = -1

        try:
            metrics["davies_bouldin_score"] = davies_bouldin_score(
                embeddings, cluster_labels
            )
        except:
            metrics["davies_bouldin_score"] = -1

        return metrics

    def calculate_external_metrics(self, true_labels, predicted_labels):
        """Внешние метрики (требуют эталонную разметку)"""
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            homogeneity_completeness_v_measure,
        )

        true_encoder = LabelEncoder()
        pred_encoder = LabelEncoder()

        true_encoded = true_encoder.fit_transform(true_labels)
        pred_encoded = pred_encoder.fit_transform(predicted_labels)

        metrics = {}

        metrics["adjusted_rand_score"] = adjusted_rand_score(true_encoded, pred_encoded)
        metrics["normalized_mutual_info"] = normalized_mutual_info_score(
            true_encoded, pred_encoded
        )

        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            true_encoded, pred_encoded
        )
        metrics["homogeneity"] = homogeneity
        metrics["completeness"] = completeness
        metrics["v_measure"] = v_measure

        return metrics

    def calculate_cluster_stats(self, cluster_labels, texts):
        """Статистика по кластерам"""
        cluster_sizes = Counter(cluster_labels)
        text_lengths = [len(text) for text in texts]

        stats = {
            "n_clusters": len(cluster_sizes),
            "total_samples": len(texts),
            "avg_cluster_size": np.mean(list(cluster_sizes.values())),
            "std_cluster_size": np.std(list(cluster_sizes.values())),
            "min_cluster_size": min(cluster_sizes.values()),
            "max_cluster_size": max(cluster_sizes.values()),
            "avg_text_length": np.mean(text_lengths),
            "std_text_length": np.std(text_lengths),
        }

        if stats["avg_cluster_size"] > 0:
            stats["cluster_size_cv"] = (
                stats["std_cluster_size"] / stats["avg_cluster_size"]
            )
        else:
            stats["cluster_size_cv"] = 0

        return stats

    def evaluate_clustering_quality(
        self, texts, cluster_labels, true_labels=None, embeddings=None
    ):
        """Комплексная оценка качества кластеризации"""
        results = {}

        results["internal_metrics"] = self.calculate_internal_metrics(
            texts, cluster_labels, embeddings
        )
        results["cluster_stats"] = self.calculate_cluster_stats(cluster_labels, texts)

        if true_labels is not None and len(true_labels) == len(cluster_labels):
            results["external_metrics"] = self.calculate_external_metrics(
                true_labels, cluster_labels
            )

        return results
