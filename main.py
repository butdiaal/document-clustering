import os
import argparse
from collections import defaultdict
from file_read import read_all_files, process_all_documents
from strateg.splitting import ParagraphSplittingStrategy, SentenceSplittingStrategy, SectionSplittingStrategy
from strateg.feature import TFIDFStrategy, BERTStrategy
from strateg.clustering import KMeansClusteringStrategy, DBSCANClusteringStrategy


def save_clustered_paragraphs(clustered_data, output_dir):
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
    parser.add_argument('--splitting', '-s', default='paragraph',
                       choices=['paragraph', 'sentence', 'section'],
                       help='Стратегия разбиения')
    parser.add_argument('--features', '-f', default='tfidf',
                        choices=['tfidf', 'bert'],
                        help='Стратегия признаков')
    parser.add_argument('--clustering', '-c', default='kmeans',
                       choices=['kmeans', 'dbscan'],
                       help='Стратегия кластеризации')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    texts, filenames = read_all_files(args.input)

    if args.splitting == 'paragraph':
        splitting_strategy = ParagraphSplittingStrategy()
    elif args.splitting == 'sentence':
        splitting_strategy = SentenceSplittingStrategy()
    elif args.splitting == 'section':
        splitting_strategy = SectionSplittingStrategy()

    if args.features == 'tfidf':
        feature_strategy = TFIDFStrategy()
    elif args.features == 'bert':
        feature_strategy = BERTStrategy()

    if args.clustering == 'kmeans':
        clustering_strategy = KMeansClusteringStrategy()
    elif args.clustering == 'dbscan':
        clustering_strategy = DBSCANClusteringStrategy()

    if not texts:
        return

    clustered_data, total_fragments = process_all_documents(
        texts, filenames, splitting_strategy, feature_strategy, clustering_strategy
    )

    save_clustered_paragraphs(clustered_data, args.output)

if __name__ == "__main__":
    main()