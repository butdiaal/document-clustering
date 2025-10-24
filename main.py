import os
import argparse
import logging
from collections import defaultdict
from file_read import read_all_files, process_all_documents
from strategy.splitting import (
    ParagraphSplittingStrategy,
    SentenceSplittingStrategy,
    SectionSplittingStrategy,
    SemanticSplittingStrategy,
    CombinedSplittingStrategy,
)
from strategy.feature import TFIDFStrategy, BERTStrategy
from metrics.test import evaluate_results
from strategy.clustering import (
    DBSCANClusteringStrategy,
    HDBSCANClusteringStrategy,
    HierarchicalClusteringStrategy,
    SemanticClusteringStrategy,
)


def save_clustered_paragraphs(clustered_data, output_dir):
    """Сохранение абзацев по тематикам"""

    for cluster_id, paragraphs_data in clustered_data.items():
        topic_dir = os.path.join(output_dir, f"topic_{cluster_id + 1}")
        os.makedirs(topic_dir, exist_ok=True)

        paragraphs_by_file = defaultdict(list)
        for para_data in paragraphs_data:
            paragraphs_by_file[para_data["source"]].append(para_data)

        for filename, file_paragraphs in paragraphs_by_file.items():
            base_name = filename.replace(".txt", "")
            output_file = os.path.join(topic_dir, f"{base_name}.txt")

            with open(output_file, "w", encoding="utf-8") as f:
                for para_data in file_paragraphs:
                    f.write(para_data["text"])


def main():
    parser = argparse.ArgumentParser(
        description="Кластеризация документов по тематикам"
    )
    parser.add_argument("--input", "-i", default="data", help="Входная директория")
    parser.add_argument(
        "--output", "-o", default="result_data", help="Выходная директория"
    )
    parser.add_argument(
        "--splitting",
        "-s",
        default="combined",
        choices=["paragraph", "sentence", "section", "combined", "semantic"],
        help="Стратегия разбиения",
    )
    parser.add_argument(
        "--features",
        "-f",
        default="bert",
        choices=["tfidf", "bert"],
        help="Стратегия признаков",
    )
    parser.add_argument(
        "--clustering",
        "-c",
        default="semantic",
        choices=["dbscan", "hdbscan", "hierarchical", "semantic"],
        help="Стратегия кластеризации",
    )

    args = parser.parse_args()

    if args.splitting == "combined":
        if args.features == "tfidf":
            logging.warning("Для комбинированного разбиения признак 'bert'")
            args.features = "bert"
        if args.clustering == "hierarchical":
            logging.warning("Для комбинированного разбиения кластеризация 'semantic'")
            args.clustering = "semantic"

    if args.splitting == "semantic":
        if args.features == "tfidf":
            logging.warning("Для семантического разбиения признак 'bert'")
            args.features = "bert"

    os.makedirs(args.output, exist_ok=True)
    texts, filenames = read_all_files(args.input)

    if args.splitting == "paragraph":
        splitting_strategy = ParagraphSplittingStrategy()
    elif args.splitting == "sentence":
        splitting_strategy = SentenceSplittingStrategy()
    elif args.splitting == "section":
        splitting_strategy = SectionSplittingStrategy()
    elif args.splitting == "combined":
        splitting_strategy = CombinedSplittingStrategy()
    elif args.splitting == "semantic":
        splitting_strategy = SemanticSplittingStrategy()

    if args.features == "tfidf":
        feature_strategy = TFIDFStrategy()
    elif args.features == "bert":
        feature_strategy = BERTStrategy()

    if args.clustering == "dbscan":
        clustering_strategy = DBSCANClusteringStrategy()
    elif args.clustering == "hdbscan":
        clustering_strategy = HDBSCANClusteringStrategy()
    elif args.clustering == "hierarchical":
        clustering_strategy = HierarchicalClusteringStrategy()
    elif args.clustering == "semantic":
        clustering_strategy = SemanticClusteringStrategy()

    if not texts:
        return

    clustered_data, total_fragments = process_all_documents(
        texts, filenames, splitting_strategy, feature_strategy, clustering_strategy
    )

    save_clustered_paragraphs(clustered_data, args.output)
    evaluate_results(clustered_data, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
