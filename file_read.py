import os
from collections import defaultdict


def save_clustered_paragraphs(clustered_data, output_dir):
    """Сохранение абзацев по тематикам"""

    import os
    from collections import defaultdict

    documents_themes = defaultdict(lambda: defaultdict(list))

    for cluster_id, paragraphs_data in clustered_data.items():
        for para_data in paragraphs_data:
            source_file = para_data["source"]
            theme_id = cluster_id
            documents_themes[source_file][theme_id].append(para_data["text"])

    for doc_index, (filename, themes_data) in enumerate(documents_themes.items(), 1):
        doc_folder_name = f"{filename.replace('.txt', '')}"
        doc_folder = os.path.join(output_dir, doc_folder_name)
        os.makedirs(doc_folder, exist_ok=True)

        for theme_index, (theme_id, theme_texts) in enumerate(themes_data.items(), 1):
            theme_file = os.path.join(doc_folder, f"theme_{theme_index:02d}.txt")

            with open(theme_file, "w", encoding="utf-8") as f:
                for text in theme_texts:
                    f.write(text)
                    f.write("\n\n")


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
    texts,
    filenames,
    splitting_strategy,
    feature_strategy,
    clustering_strategy,
    output_dir,
):
    """Обработка всех документов"""

    total_fragments = 0

    for text, filename in zip(texts, filenames):
        doc_clustered_data = defaultdict(list)
        doc_fragments_count = 0

        fragments = splitting_strategy.split(text)

        if len(fragments) <= 1:
            cluster_id = f"{filename}_0"
            for i, fragment in enumerate(fragments):
                doc_clustered_data[cluster_id].append(
                    {"text": fragment, "source": filename, "fragment_id": i}
                )
            doc_fragments_count = len(fragments)
        else:
            features = feature_strategy.transform(fragments)
            cluster_labels = clustering_strategy.cluster(features)

            for i, (fragment, cluster_id) in enumerate(zip(fragments, cluster_labels)):
                global_cluster_id = f"{filename}_{cluster_id}"
                doc_clustered_data[global_cluster_id].append(
                    {"text": fragment, "source": filename, "fragment_id": i}
                )
            doc_fragments_count = len(fragments)

        save_clustered_paragraphs(doc_clustered_data, output_dir)

        total_fragments += doc_fragments_count

    return defaultdict(list), total_fragments


def process_combined(
    texts,
    filenames,
    splitting_strategy,
    feature_strategy,
    clustering_strategy,
    output_dir,
):
    """Обработка для комбинированного метода"""

    total_fragments = 0

    for text, filename in zip(texts, filenames):
        doc_clustered_data = defaultdict(list)
        doc_fragments_count = 0

        paragraphs, sentences_data = splitting_strategy.split(text)

        for sent_data in sentences_data:
            sent_data["source_file"] = filename

        final_paragraphs = clustering_strategy.cluster_paragraphs_sentences(
            paragraphs, sentences_data
        )

        for i, meta_para in enumerate(final_paragraphs):
            cluster_id = i
            doc_clustered_data[cluster_id].append(
                {"text": meta_para, "source": filename, "fragment_id": i}
            )
            doc_fragments_count += 1

        save_clustered_paragraphs(doc_clustered_data, output_dir)

        total_fragments += doc_fragments_count

    return defaultdict(list), total_fragments
