import os
import argparse
from collections import defaultdict
from file_read import read_all_files, process_all_documents


def save_clustered_paragraphs(clustered_data, output_dir, filenames):
    """Сохранение абзацев по тематикам"""

    for cluster_id, paragraphs_data in clustered_data.items():
        topic_dir = os.path.join(output_dir, f"topic_{cluster_id + 1}")
        os.makedirs(topic_dir, exist_ok=True)

        paragraphs_by_file = defaultdict(list)
        for para_data in paragraphs_data:
            paragraphs_by_file[para_data['source']].append(para_data)

        for filename, file_paragraphs in paragraphs_by_file.items():
            base_name = filename.replace('.txt', '')
            output_file = os.path.join(topic_dir, f"{base_name}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                for para_data in file_paragraphs:
                    f.write(para_data['text'])



def main():
    parser = argparse.ArgumentParser(description='Кластеризация документов по тематикам')
    parser.add_argument('--input', '-i', default='data',
                        help='Входная директория')
    parser.add_argument('--output', '-o', default='result_data',
                        help='Выходная директория')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    texts, filenames = read_all_files(args.input)

    if not texts:
        return

    clustered_data, total_paragraphs = process_all_documents(texts, filenames)
    save_clustered_paragraphs(clustered_data, args.output, filenames)


if __name__ == "__main__":
    main()