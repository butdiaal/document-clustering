import logging
import os
import numpy as np
from collections import Counter
from metric.metric import ClusteringMetrics
from sentence_transformers import SentenceTransformer
from config import BERT_MODEL_NAME

try:
    from data.manual_labeling import (
        reference_labels,
        theme_names,
        get_reference_mapping,
        get_expected_themes_count,
    )

    HAS_REFERENCE = True
except ImportError:
    HAS_REFERENCE = False


class ReferenceEvaluator:
    """Класс для сравнения с эталонной разметкой"""

    def __init__(self):
        self.model = SentenceTransformer(BERT_MODEL_NAME)
        if HAS_REFERENCE:
            self.reference_mapping = get_reference_mapping()
        else:
            self.reference_mapping = {}

    def create_reference_embeddings(self):
        """Создает эмбеддинги для эталонных тем"""
        if not HAS_REFERENCE:
            return {}

        theme_embeddings = {}
        for (filename, theme_id), theme_info in self.reference_mapping.items():
            theme_text = f"{theme_info['name']} {' '.join(theme_info['keywords'])}"
            embedding = self.model.encode(theme_text, normalize_embeddings=True)
            theme_embeddings[(filename, theme_id)] = embedding
        return theme_embeddings

    def match_clusters_to_reference(
        self, clustered_data, all_fragments, fragment_sources
    ):
        """Сопоставляет алгоритмические кластеры с эталонными темами"""
        if not HAS_REFERENCE:
            return {}

        reference_embeddings = self.create_reference_embeddings()
        cluster_embeddings = {}
        cluster_texts = {}

        for cluster_id, fragments in clustered_data.items():
            cluster_text = " ".join([f["text"] for f in fragments[:5]])
            embedding = self.model.encode(cluster_text, normalize_embeddings=True)
            cluster_embeddings[cluster_id] = embedding
            cluster_texts[cluster_id] = cluster_text[:200]

        matches = {}
        for cluster_id, cluster_emb in cluster_embeddings.items():
            best_match = None
            best_similarity = -1

            for ref_key, ref_emb in reference_embeddings.items():
                similarity = np.dot(cluster_emb, ref_emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = ref_key

            if best_similarity > 0.6:
                matches[cluster_id] = {
                    "reference_theme": best_match,
                    "similarity": best_similarity,
                    "theme_name": theme_names.get(
                        best_match[1], f"ТЕМА_{best_match[1]}"
                    ),
                    "cluster_preview": cluster_texts[cluster_id],
                }

        return matches


def evaluate_results(
    clustered_data,
    output_dir,
    all_fragments=None,
    fragment_sources=None,
    feature_strategy=None,
):
    """Оценка результатов кластеризации"""

    theme_folders = [f for f in os.listdir(output_dir) if f.startswith("topic_")]
    algorithm_themes = len(theme_folders)

    all_fragments_list = []
    all_labels = []

    for cluster_id, fragments in clustered_data.items():
        for fragment_data in fragments:
            all_fragments_list.append(fragment_data["text"])
            all_labels.append(cluster_id)

    total_fragments = len(all_fragments_list)

    logging.info(f"Алгоритм создал тем: {algorithm_themes}")
    logging.info(f"Всего фрагментов: {total_fragments}")

    for cluster_id, fragments in clustered_data.items():
        sources = Counter([f["source"] for f in fragments])
        logging.info(
            f"Тема {cluster_id + 1}: {len(fragments)} фрагментов из {len(sources)} документов"
        )

    if all_fragments_list and feature_strategy:
        try:
            metrics_calculator = ClusteringMetrics()
            features = feature_strategy.transform(all_fragments_list)
            metrics_results = metrics_calculator.evaluate_clustering_quality(
                texts=all_fragments_list,
                cluster_labels=all_labels,
                embeddings=features if hasattr(features, "shape") else None,
            )

            _evaluate_by_metrics(metrics_results, algorithm_themes)

        except Exception as e:
            logging.error(f"Ошибка при расчете метрик: {e}")

    if HAS_REFERENCE:
        evaluator = ReferenceEvaluator()
        matches = evaluator.match_clusters_to_reference(
            clustered_data, all_fragments_list, fragment_sources
        )

        logging.info(f"Найдено соответствий: {len(matches)} из {algorithm_themes}")

        expected_themes = get_expected_themes_count()
        logging.info(f"Ожидаемое количество тем: {expected_themes}")

        for cluster_id, match_info in matches.items():
            logging.info(
                f" Кластер {cluster_id + 1} - {match_info['theme_name']} (схожесть: {match_info['similarity']:.3f})"
            )

        coverage_ratio = len(matches) / expected_themes if expected_themes > 0 else 0
        if coverage_ratio >= 0.7:
            logging.info(f"Хорошее покрытие эталона: {coverage_ratio:.1%}")
        elif coverage_ratio >= 0.5:
            logging.info(f"Умеренное покрытие эталона: {coverage_ratio:.1%}")
        else:
            logging.warning(f"Низкое покрытие эталона: {coverage_ratio:.1%}")

        expected_range = (expected_themes - 2, expected_themes + 3)
        if expected_range[0] <= algorithm_themes <= expected_range[1]:
            logging.info(f"Количество тем соответствует ожидаемому ({expected_themes})")
        elif algorithm_themes < expected_range[0]:
            logging.warning(
                f"Слишком мало тем ({algorithm_themes} вместо {expected_themes})"
            )
        else:
            logging.warning(
                f"Слишком много тем ({algorithm_themes} вместо {expected_themes})"
            )
    else:
        logging.info("Эталонная разметка не найдена")


def _evaluate_by_metrics(metrics_results, n_clusters):
    """Оценивает качество кластеризации по метрикам"""

    internal_metrics = metrics_results.get("internal_metrics", {})
    cluster_stats = metrics_results.get("cluster_stats", {})

    silhouette = internal_metrics.get("silhouette_score", -1)
    if silhouette > 0.5:
        logging.info(f"Отличный силуэтный скор: {silhouette:.3f}")
    elif silhouette > 0.25:
        logging.info(f"Удовлетворительный силуэтный скор: {silhouette:.3f}")
    elif silhouette >= 0:
        logging.warning(f"Слабый силуэтный скор: {silhouette:.3f}")
    else:
        logging.warning("Не удалось рассчитать силуэтный скор")

    db_score = internal_metrics.get("davies_bouldin_score", float("inf"))
    if db_score < 0.7:
        logging.info(f"Отличный Davies-Bouldin: {db_score:.3f}")
    elif db_score < 1.0:
        logging.info(f"Удовлетворительный Davies-Bouldin: {db_score:.3f}")
    elif db_score < float("inf"):
        logging.warning(f"Слабый Davies-Bouldin: {db_score:.3f}")
    else:
        logging.warning("Не удалось рассчитать Davies-Bouldin")

    cv = cluster_stats.get("cluster_size_cv", 1)
    if cv < 0.5:
        logging.info(f"Хорошая сбалансированность кластеров (CV: {cv:.3f})")
    elif cv < 1.0:
        logging.info(f"Умеренная сбалансированность кластеров (CV: {cv:.3f})")
    else:
        logging.warning(f"Несбалансированные кластеры (CV: {cv:.3f})")

    if silhouette > 0.3 and db_score < 1.0 and cv < 1.0:
        logging.info("✓ Общее качество кластеризации: хорошее")
    elif silhouette > 0.1 and db_score < 1.5:
        logging.info("Общее качество кластеризации: удовлетворительное")
    else:
        logging.warning("Общее качество кластеризации: низкое")


def analyze_cluster_quality(clustered_data):
    """Детальный анализ качества кластеров"""
    quality_report = {}

    for cluster_id, fragments in clustered_data.items():
        sources = set(f["source"] for f in fragments)
        fragment_count = len(fragments)
        avg_length = np.mean([len(f["text"]) for f in fragments])

        quality_report[cluster_id] = {
            "fragment_count": fragment_count,
            "source_diversity": len(sources),
            "avg_length": avg_length,
            "sources": list(sources),
        }

    return quality_report
