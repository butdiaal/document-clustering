import os


SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.7))
CONTEXT_WINDOW = int(os.getenv('CONTEXT_WINDOW', 3))
MIN_SENTENCES_PER_CLUSTER = int(os.getenv('MIN_SENTENCES_PER_CLUSTER', 2))

MIN_LENGTH = int(os.getenv('MIN_LENGTH', 50))
MAX_PARAGRAPH_LENGTH = int(os.getenv('MAX_PARAGRAPH_LENGTH', 1000))
SENTENCE_GROUP_SIZE = int(os.getenv('SENTENCE_GROUP_SIZE', 3))

MAX_CLUSTERS = int(os.getenv('MAX_CLUSTERS', 10))
MIN_CLUSTER_SIZE = int(os.getenv('MIN_CLUSTER_SIZE', 2))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

BERT_MODEL_NAME = os.getenv('BERT_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')