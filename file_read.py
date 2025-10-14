import os
from collections import defaultdict


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


def process_all_documents(texts, filenames, splitting_strategy):
    """Обработка всех документов"""

    all_fragments = []
    fragment_sources = []

    for text, filename in zip(texts, filenames):
        fragments = splitting_strategy.split(text)
        all_fragments.extend(fragments)
        fragment_sources.extend([filename] * len(fragments))

    clustered_data = defaultdict(list)
    for i, (fragments, source, cluster_id) in enumerate(zip(all_fragments, fragment_sources)):
        clustered_data[cluster_id].append({
            'text': fragments,
            'source': source,
            'paragraph_id': i
        })

    return clustered_data, len(all_fragments)