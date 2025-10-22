import logging
import os
from data.manual_labeling import reference_labels
from collections import Counter


def evaluate_results(clustered_data, output_dir):
    """Оценка результатов кластеризации"""

    theme_folders = [f for f in os.listdir(output_dir) if f.startswith('topic_')]
    algorithm_themes = len(theme_folders)

    total_fragments = sum(len(fragments) for fragments in clustered_data.values())
    logging.info(f"Алгоритм создал тем: {algorithm_themes}")
    logging.info(f"Всего фрагментов: {total_fragments}")

    for cluster_id, fragments in clustered_data.items():
        sources = Counter([f['source'] for f in fragments])
        logging.info(f"Тема {cluster_id + 1}: {len(fragments)} фрагментов из {len(sources)} документов")

    reference = reference_labels
    total_reference_themes = sum(len(themes) for themes in reference.values())
    logging.info(f"Эталон содержит: {total_reference_themes} тематических групп")

    if algorithm_themes >= 8:
        logging.info("Количество тем соответствует")
    else:
        logging.warning("Количество тем меньше")