import os
import argparse
import logging
from collections import defaultdict
from file_read import read_all_files, process_all_documents, process_combined
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

    if args.splitting == "combined":
        splitting_strategy = CombinedSplittingStrategy()
        feature_strategy = BERTStrategy()
        clustering_strategy = SemanticClusteringStrategy()
        clustered_data, total_fragments = process_combined(
            texts,
            filenames,
            splitting_strategy,
            feature_strategy,
            clustering_strategy,
            args.output,
        )
    else:
        if args.splitting == "paragraph":
            splitting_strategy = ParagraphSplittingStrategy()
        elif args.splitting == "sentence":
            splitting_strategy = SentenceSplittingStrategy()
        elif args.splitting == "section":
            splitting_strategy = SectionSplittingStrategy()
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
            clustering_strategy = SemanticClusteringStrategy(
                eps=args.eps, min_samples=args.min_samples
            )

        clustered_data, total_fragments = process_all_documents(
            texts,
            filenames,
            splitting_strategy,
            feature_strategy,
            clustering_strategy,
            args.output,
        )

    # evaluate_results(clustered_data, args.output)


if __name__ == "__main__":
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
