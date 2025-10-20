from abc import ABC, abstractmethod
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from config import *


class SplittingStrategy(ABC):
    """Абстрактный класс стратегии разбиения документа"""

    @abstractmethod
    def split(self, text):
        """Разбивает текст на фрагменты"""
        pass


class ParagraphSplittingStrategy(SplittingStrategy):
    """Стратегия разбиения по абзацам"""

    def __init__(self, min_length=50):
        self.min_length = min_length or MIN_LENGTH

    def split(self, text):
        """Разбивает текст на абзацы"""
        paragraphs = []
        raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for paragraph in raw_paragraphs:
            if len(paragraph) > MAX_PARAGRAPH_LENGTH:
                sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
                for i in range(0, len(sentences), 4):
                    chunk = '. '.join(sentences[i:i + 4]) + '.'
                    if len(chunk) >= self.min_length:
                        paragraphs.append(chunk)
            elif len(paragraph) >= self.min_length:
                paragraphs.append(paragraph)

        return paragraphs


class SentenceSplittingStrategy(SplittingStrategy):
    """Стратегия разбиения по предложениям с группировкой"""

    def __init__(self, group_size=3, min_length=30):
        self.group_size = group_size or SENTENCE_GROUP_SIZE
        self.min_length = min_length or MIN_LENGTH

    def split(self, text):
        """Разбивает текст на группы предложений"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        paragraphs = []
        for i in range(0, len(sentences), self.group_size):
            chunk = '. '.join(sentences[i:i + self.group_size]) + '.'
            if len(chunk) >= self.min_length:
                paragraphs.append(chunk)

        return paragraphs


class SectionSplittingStrategy(SplittingStrategy):
    """Стратегия разбиения по разделам"""

    def split(self, text):
        """Разбивает текст по структурным элементам"""
        pattern = r'(Статья\s+\d+\.|\d+\.\d+\.|\d+\.\s+[А-Я]|Раздел\s+[IVXLCDM]+)'
        sections = re.split(pattern, text)

        paragraphs = []
        current_section = ""

        for i, part in enumerate(sections):
            if re.match(pattern, part) and current_section:
                paragraphs.append(current_section.strip())
                current_section = part + " "
            else:
                current_section += part

        if current_section:
            paragraphs.append(current_section.strip())

        return [p for p in paragraphs if len(p) > 50]


class CombinedSplittingStrategy(SplittingStrategy):
    """Комбинированное разбиение: предложения + кластеризация + мета-абзацы"""

    def __init__(self, min_sentences_per_cluster=2):
        self.min_sentences_per_cluster =  min_sentences_per_cluster or MIN_SENTENCES_PER_CLUSTER
        self.model = SentenceTransformer(BERT_MODEL_NAME)

    def split(self, text):
        """Разбиение по абзацам и предложениям"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        sentences_with_context = []
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
            for sent_idx, sentence in enumerate(sentences):
                sentences_with_context.append({
                    'text': sentence,
                    'paragraph_id': para_idx,
                    'sentence_id': sent_idx,
                    'original_paragraph': paragraph
                })

        if len(sentences_with_context) <= 1:
            return [text]

        """Кластеризация предложений"""
        sentence_texts = [s['text'] for s in sentences_with_context]
        embeddings = self.model.encode(sentence_texts, normalize_embeddings=True)

        n_clusters = min(MAX_CLUSTERS, max(MIN_CLUSTER_SIZE, len(sentences_with_context) // 3))
        kmeans =  KMeans(n_clusters=n_clusters, random_state=CLUSTERING_RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        """Кластеры, где больше 1 предложения, абзацы объединяем в 1 мета-абзац"""
        clusters = {}
        for sentence_data, cluster_id in zip(sentences_with_context, cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(sentence_data)

        meta_paragraphs = []
        used_paragraphs = set()

        for cluster_id, cluster_sentences in clusters.items():
            if len(cluster_sentences) >= self.min_sentences_per_cluster:
                cluster_sentences.sort(key=lambda x: (x['paragraph_id'], x['sentence_id']))
                meta_text = ' '.join([s['text'] for s in cluster_sentences])
                meta_paragraphs.append(meta_text)

                for s in cluster_sentences:
                    used_paragraphs.add(s['original_paragraph'])

        for paragraph in paragraphs:
            if paragraph not in used_paragraphs:
                meta_paragraphs.append(paragraph)

        return meta_paragraphs if meta_paragraphs else [text]


class SemanticSplittingStrategy(SplittingStrategy):
    """Семантическое разбиение с учетом структуры и смысла"""

    def __init__(self, min_length=50):
        self.min_length = min_length or MIN_LENGTH
        self.similarity_threshold =  SIMILARITY_THRESHOLD
        self.context_window = CONTEXT_WINDOW
        self.model = SentenceTransformer(BERT_MODEL_NAME)

    def split(self, text):
        """Алгоритм сплита: приоритетно по \\n\\n, затем по семантическим разрывам"""
        segments = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        for paragraph in paragraphs:
            if self.is_semantically_coherent(paragraph):
                segments.append(paragraph)
            else:
                sub_segments = self.split_by_semantic_breaks(paragraph)
                segments.extend(sub_segments)

        return [s for s in segments if len(s) >= self.min_length]

    def is_semantically_coherent(self, text):
        """Проверяет, что все предложения в тексте на одну тему"""
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return True

        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        similarities = []
        for i in range(1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[i - 1])
            similarities.append(similarity)

        return min(similarities) > self.similarity_threshold if similarities else True

    def split_by_semantic_breaks(self, text):
        """Разбивает текст по семантическим разрывам"""
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return [text]

        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        boundaries = []
        for i in range(1, len(sentences)):
            similarity_prev = np.dot(embeddings[i], embeddings[i - 1])

            start_idx = max(0, i - self.context_window)
            context_embedding = np.mean(embeddings[start_idx:i], axis=0)
            similarity_context = np.dot(embeddings[i], context_embedding)

            if similarity_prev < self.similarity_threshold or similarity_context < self.similarity_threshold:
                boundaries.append(i)

        segments = []
        start_idx = 0

        for boundary in boundaries:
            segment_sentences = sentences[start_idx:boundary]
            if segment_sentences:
                segment = ' '.join(segment_sentences)
                segments.append(segment)
            start_idx = boundary

        if start_idx < len(sentences):
            segment = ' '.join(sentences[start_idx:])
            segments.append(segment)

        return segments