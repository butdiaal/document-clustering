import os
from collections import defaultdict
import numpy as np
from strategy.clustering import DBSCANClusteringStrategy


def read_all_files(input_dir):
    """Чтение всех файлов из директории"""
    texts = []
    filenames = []

    for file in os.listdir(input_dir):
        if file.endswith(".txt"):
            with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(content)
                filenames.append(file)

    return texts, filenames


def extract_paragraphs(texts, filenames, splitting_strategy):
    """Обработка всех документов"""

    all_fragments = []
    fragment_sources = []

    for text, filename in zip(texts, filenames):
        fragments = splitting_strategy.split(text)
        all_fragments.extend(fragments)
        fragment_sources.extend([filename] * len(fragments))

    return all_fragments, fragment_sources


def process_all_documents(
    texts, filenames, splitting_strategy, feature_strategy, clustering_strategy
):
    """Обработка всех документов"""

    all_fragments, fragment_sources = extract_paragraphs(
        texts, filenames, splitting_strategy
    )

    features = feature_strategy.transform(all_fragments)
    cluster_labels = clustering_strategy.cluster(features)

    clustered_data = defaultdict(list)
    for i, (fragment, source, cluster_id) in enumerate(
        zip(all_fragments, fragment_sources, cluster_labels)
    ):
        clustered_data[cluster_id].append(
            {"text": fragment, "source": source, "fragment_id": i}
        )

    return clustered_data, len(all_fragments)


def process_combined(texts, filenames, splitting_strategy, feature_strategy, clustering_strategy):
    """Обработка для комбинированного метода"""

    clustered_data = defaultdict(list)
    total_fragments = 0

    for text, filename in zip(texts, filenames):
        paragraph_all, sentence_all = splitting_strategy.split(text)

        meta_paragraphs = clustering_strategy.cluster_paragraphs_sentences(paragraph_all, sentence_all)

        if meta_paragraphs:
            features = feature_strategy.transform(meta_paragraphs)

            cluster_labels = clustering_strategy.cluster(features)

            for i, (meta_para, cluster_id) in enumerate(zip(meta_paragraphs, cluster_labels)):
                clustered_data[cluster_id].append({
                    "text": meta_para,
                    "source": filename,
                    "fragment_id": total_fragments
                })
                total_fragments += 1

    return clustered_data, total_fragments