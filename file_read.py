import os
from collections import defaultdict
from strateg.splitting import ParagraphSplittingStrategy


def read_all_files(input_dir):
    """Чтение всех файлов из директории"""
    texts = []
    filenames = []

    for file in os.listdir(input_dir):
        if file.endswith('.txt'):
            with open(os.path.join(input_dir, file), 'r', encoding='utf-8') as f:
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


def process_all_documents(texts, filenames, splitting_strategy, feature_strategy, clustering_strategy):
    """Обработка всех документов"""

    all_fragments, fragment_sources = extract_paragraphs(texts, filenames, splitting_strategy)

    features = feature_strategy.transform(all_fragments)

    cluster_labels = clustering_strategy.cluster(features)

    clustered_data = defaultdict(list)
    for i, (fragment, source, cluster_id) in enumerate(zip(all_fragments, fragment_sources, cluster_labels)):
        clustered_data[cluster_id].append({
            'text': fragment,
            'source': source,
            'fragment_id': i
        })

    return clustered_data, len(all_fragments)
