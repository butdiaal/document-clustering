from abc import ABC, abstractmethod
import re
from scipy.signal import argrelextrema
import numpy as np
from sentence_transformers import SentenceTransformer


class SplittingStrategy(ABC):
    """Абстрактный класс стратегии разбиения документа"""

    @abstractmethod
    def split(self, text):
        """Разбивает текст на фрагменты"""
        pass


class ParagraphSplittingStrategy(SplittingStrategy):
    """Стратегия разбиения по абзацам"""

    def __init__(self, min_length=50):
        self.min_length = min_length

    def split(self, text):
        """Разбивает текст на абзацы"""
        paragraphs = []
        raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        for paragraph in raw_paragraphs:
            if len(paragraph) > 1000:
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
        self.group_size = group_size
        self.min_length = min_length

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


class SemanticSplittingStrategy(SplittingStrategy):
    """Семантическое разбиение"""

    def __init__(self, min_length=50):
        self.min_length = min_length
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def split(self, text):
        """Разбивает текст на семантические сегменты"""
        lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10]

        if len(lines) <= 1:
            return [' '.join(lines)] if lines else []

        embeddings = self.model.encode(lines, normalize_embeddings=True)

        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        similarities = np.array(similarities)

        boundaries = self._find_boundaries(similarities)

        segments = self._create_segments(lines, boundaries)

        return [s for s in segments if len(s) >= self.min_length]

    def _find_boundaries(self, similarities):
        """Поиск границ через локальные минимумы"""
        if len(similarities) < 3:
            return []

        minima = argrelextrema(similarities, np.less)[0]

        if len(minima) == 0:
            return []

        avg_similarity = np.mean(similarities)
        boundaries = []

        for min_idx in minima:
            if similarities[min_idx] < avg_similarity:
                boundaries.append(min_idx + 1)  # +1 как в том коде

        return boundaries

    def _create_segments(self, lines, boundaries):
        """Собирает сегменты по границам"""
        if not boundaries:
            return [' '.join(lines)]

        segments = []
        start = 0

        for boundary in boundaries:
            segment_lines = lines[start:boundary]
            if segment_lines:
                segment = ' '.join(segment_lines)
                segments.append(segment)
            start = boundary

        if start < len(lines):
            segment = ' '.join(lines[start:])
            segments.append(segment)

        return segments