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
    """Исправленная обработка для комбинированного метода"""

    all_paragraphs = []
    all_sentences_data = []
    paragraph_info = []

    for text, filename in zip(texts, filenames):
        paragraphs, sentences_data = splitting_strategy.split(text)

        for para_idx, paragraph in enumerate(paragraphs):
            all_paragraphs.append(paragraph)
            paragraph_info.append({
                "text": paragraph,
                "source": filename,
                "global_idx": len(all_paragraphs) - 1
            })

        for sent_data in sentences_data:
            sent_data["source_file"] = filename
            all_sentences_data.append(sent_data)

    final_paragraphs, paragraph_mapping = clustering_strategy.cluster_paragraphs_sentences(
        all_paragraphs, all_sentences_data
    )

    if final_paragraphs:
        features = feature_strategy.transform(final_paragraphs)
        cluster_labels = clustering_strategy.cluster(features)
    else:
        cluster_labels = []

    clustered_data = defaultdict(list)

    for i, (meta_para, cluster_id) in enumerate(zip(final_paragraphs, cluster_labels)):
        if i in paragraph_mapping:
            source_files = set()

            for para_idx in paragraph_mapping[i]:
                if para_idx < len(paragraph_info):
                    source_files.add(paragraph_info[para_idx]["source"])

            for source_file in source_files:
                clustered_data[cluster_id].append({
                    "text": meta_para,
                    "source": source_file,
                    "fragment_id": i
                })

    return clustered_data, len(final_paragraphs)