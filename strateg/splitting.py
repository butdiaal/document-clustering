from abc import ABC, abstractmethod
import re
import numpy as np
from sentence_transformers import SentenceTransformer
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
    """Стратегия разбиения: абзацы - строки - предложения"""

    def split(self, text):
        """Разбивает текст: по абзацам, строкам и предложениям"""

        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

        refined_paragraphs = []
        for paragraph in paragraphs:
            if len(paragraph) > MAX_PARAGRAPH_LENGTH:
                lines = [line.strip() for line in paragraph.split('\n') if line.strip()]
                for line in lines:
                    if len(line) > MAX_PARAGRAPH_LENGTH:
                        sentences = [s.strip() + '.' for s in line.split('.') if s.strip()]
                        for i in range(0, len(sentences), SENTENCE_GROUP_SIZE):
                            chunk = ' '.join(sentences[i:i + SENTENCE_GROUP_SIZE])
                            if len(chunk) >= MIN_LENGTH:
                                refined_paragraphs.append(chunk)
                    elif len(line) >= MIN_LENGTH:
                        refined_paragraphs.append(line)
            elif len(paragraph) >= MIN_LENGTH:
                refined_paragraphs.append(paragraph)

        return refined_paragraphs if refined_paragraphs else [text]


class SemanticSplittingStrategy(SplittingStrategy):
    """Семантическое разбиение с объединением соседних абзацев через анализ ряда схожестей"""

    def __init__(self):
        self.model = SentenceTransformer(BERT_MODEL_NAME)
        self.similarity_threshold = SIMILARITY_THRESHOLD

    def split(self, text):
        """Разбивает текст с анализом схожести между соседними абзацами"""
        raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

        if len(raw_paragraphs) <= 1:
            return self.split_large_paragraphs([text])

        similarity_series = self.build_similarity_series(raw_paragraphs)
        merge_decisions = self.analyze_similarity_breaks(similarity_series)
        processed_paragraphs = self.merge_paragraphs_by_decisions(raw_paragraphs, merge_decisions)
        final_paragraphs = self.split_large_paragraphs(processed_paragraphs)

        return [p for p in final_paragraphs if len(p) >= MIN_LENGTH]

    def build_similarity_series(self, paragraphs):
        """Строит ряд схожестей между соседними абзацами"""
        similarities = []

        for i in range(1, len(paragraphs)):
            prev_para = paragraphs[i - 1]
            curr_para = paragraphs[i]

            prev_elements = self.split_into_elements(prev_para)
            curr_elements = self.split_into_elements(curr_para)

            if not prev_elements or not curr_elements:
                similarities.append(0.0)
                continue

            all_elements = prev_elements + curr_elements
            embeddings = self.model.encode(all_elements, normalize_embeddings=True)

            prev_embeddings = embeddings[:len(prev_elements)]
            curr_embeddings = embeddings[len(prev_elements):]

            max_similarity = 0
            for curr_emb in curr_embeddings:
                for prev_emb in prev_embeddings:
                    similarity = np.dot(curr_emb, prev_emb)
                    max_similarity = max(max_similarity, similarity)

            similarities.append(max_similarity)

        return similarities

    def analyze_similarity_breaks(self, similarity_series):
        """Анализирует провалы в ряду схожестей через Depth Score"""
        if len(similarity_series) < 2:
            return [True] * len(similarity_series)  # Все объединяем

        merge_decisions = []

        for i in range(len(similarity_series)):
            current_similarity = similarity_series[i]

            if i == 0:
                depth = similarity_series[i + 1] - current_similarity if i + 1 < len(similarity_series) else 0
            elif i == len(similarity_series) - 1:
                depth = similarity_series[i - 1] - current_similarity
            else:
                depth = min(similarity_series[i - 1], similarity_series[i + 1]) - current_similarity

            should_merge = (depth <= 0.15 and current_similarity > self.similarity_threshold)
            merge_decisions.append(should_merge)

        return merge_decisions

    def merge_paragraphs_by_decisions(self, paragraphs, merge_decisions):
        """Объединяет абзацы на основе решений о слиянии"""
        processed = []
        current_group = [paragraphs[0]]

        for i in range(1, len(paragraphs)):
            if merge_decisions[i - 1]:
                current_group.append(paragraphs[i])
            else:
                processed.append("\n\n".join(current_group))
                current_group = [paragraphs[i]]

        processed.append("\n\n".join(current_group))

        return processed

    def split_into_elements(self, paragraph):
        """Разбивает абзац на элементы (предложения)"""
        sentences = [s.strip() + '.' for s in paragraph.split('.') if s.strip()]
        return [s for s in sentences if len(s) > 10]

    def split_large_paragraphs(self, paragraphs):
        """Разбивает слишком большие абзацы по семантическим провалам"""
        result = []
        for paragraph in paragraphs:
            if len(paragraph) <= MAX_PARAGRAPH_LENGTH:
                result.append(paragraph)
            else:
                result.extend(self.split_single_large_paragraph(paragraph))
        return result

    def split_single_large_paragraph(self, paragraph):
        """Разбивает один большой абзац по семантическим провалам"""
        elements = self.split_into_elements(paragraph)
        if len(elements) <= 1:
            return [paragraph]

        embeddings = self.model.encode(elements, normalize_embeddings=True)

        similarities = []
        for i in range(1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[i - 1])
            similarities.append(similarity)

        split_points = self.find_semantic_breaks(similarities)

        fragments = []
        start_idx = 0

        for split_idx in split_points:
            fragment_elements = elements[start_idx:split_idx + 1]
            if fragment_elements:
                fragment = ' '.join(fragment_elements)
                fragments.append(fragment)
            start_idx = split_idx + 1

        if start_idx < len(elements):
            fragment = ' '.join(elements[start_idx:])
            fragments.append(fragment)

        return fragments if fragments else [paragraph]

    def find_semantic_breaks(self, similarities):
        """Находит семантические провалы в ряду схожестей через Depth Score"""
        if len(similarities) < 3:
            return []

        breaks = []

        for i in range(1, len(similarities) - 1):

            depth = min(similarities[i - 1], similarities[i + 1]) - similarities[i]
            if depth > 0.15 and similarities[i] < self.similarity_threshold:
                breaks.append(i)

        return breaks