import os
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


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


def extract_paragraphs_from_all_documents(texts, filenames):
    """Извлечение абзацев из документов"""
    all_paragraphs = []
    paragraph_sources = []

    for text, filename in zip(texts, filenames):
        paragraphs = extract_paragraphs(text)
        all_paragraphs.extend(paragraphs)
        paragraph_sources.extend([filename] * len(paragraphs))

    return all_paragraphs, paragraph_sources


def extract_paragraphs(text, min_length=50):
    """Извлечение абзацев из текста"""
    paragraphs = []

    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    for paragraph in raw_paragraphs:
        if len(paragraph) > 1000:
            sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
            for i in range(0, len(sentences), 4):
                chunk = '. '.join(sentences[i:i + 4]) + '.'
                if len(chunk) >= min_length:
                    paragraphs.append(chunk)
        elif len(paragraph) >= min_length:
            paragraphs.append(paragraph)

    return paragraphs


def cluster_all_paragraphs(paragraphs, n_clusters=None):
    """Кластеризация всех абзацев из документов по общим темам"""
    if not paragraphs:
        return {}

    if n_clusters is None:
        n_clusters = min(8, max(2, len(paragraphs) // 3))

    if len(paragraphs) <= 1:
        return {0: paragraphs}

    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=1,
        max_df=0.8,
        stop_words=None,
        ngram_range=(1, 2)
    )

    x = vectorizer.fit_transform(paragraphs)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(x)

    return cluster_labels


def process_all_documents(texts, filenames):
    """Обработка всех документов"""

    all_paragraphs, paragraph_sources = extract_paragraphs_from_all_documents(texts, filenames)

    cluster_labels = cluster_all_paragraphs(all_paragraphs)

    clustered_data = defaultdict(list)
    for i, (paragraph, source, cluster_id) in enumerate(zip(all_paragraphs, paragraph_sources, cluster_labels)):
        clustered_data[cluster_id].append({
            'text': paragraph,
            'source': source,
            'paragraph_id': i
        })


    return clustered_data, len(all_paragraphs)