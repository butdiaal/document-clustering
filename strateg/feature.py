from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class FeatureStrategy(ABC):
    """Абстрактный класс стратегии получения признаков"""

    @abstractmethod
    def transform(self, texts):
        """Преобразует тексты в векторы признаков"""
        pass


class TFIDFStrategy(FeatureStrategy):
    """Стратегия TF-IDF"""

    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=1,
            max_df=0.8,
            stop_words=None,
            ngram_range=ngram_range
        )

    def transform(self, texts):
        return self.vectorizer.fit_transform(texts)


class BERTStrategy(FeatureStrategy):
    """Стратегия BERT"""

    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def transform(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings