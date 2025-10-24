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
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for paragraph in raw_paragraphs:
            if len(paragraph) > MAX_PARAGRAPH_LENGTH:
                sentences = [s.strip() for s in paragraph.split(".") if s.strip()]
                for i in range(0, len(sentences), 4):
                    chunk = ". ".join(sentences[i : i + 4]) + "."
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
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        paragraphs = []
        for i in range(0, len(sentences), self.group_size):
            chunk = ". ".join(sentences[i : i + self.group_size]) + "."
            if len(chunk) >= self.min_length:
                paragraphs.append(chunk)

        return paragraphs


class SectionSplittingStrategy(SplittingStrategy):
    """Стратегия разбиения по разделам"""

    def split(self, text):
        """Разбивает текст по структурным элементам"""
        pattern = r"(Статья\s+\d+\.|\d+\.\d+\.|\d+\.\s+[А-Я]|Раздел\s+[IVXLCDM]+)"
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
    """Стратегия разбиения комбинированным методом"""

    def __init__(self, use_clustering=True):
        self.use_clustering = use_clustering
        self.sentences_data = []
        self.paragraphs = []

    def split(self, text):
        """Разбивает текст на абзацы"""

        self.paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        if len(self.paragraphs) <= 1:
            return self.paragraphs

        self.sentences_data = []

        for para_idx, paragraph in enumerate(self.paragraphs):
            sentences = self._split_into_sentences(paragraph)
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    self.sentences_data.append(
                        {
                            "sentence": sentence,
                            "para_idx": para_idx,
                            "para_text": paragraph,
                        }
                    )

        return self.paragraphs

    def _split_into_sentences(self, text):
        """Разбивает текст на предложения"""
        sentences = []
        for sent in text.split("."):
            sent = sent.strip()
            if sent and len(sent) > 5:
                sentences.append(sent + ".")
        return sentences


class SemanticSplittingStrategy(SplittingStrategy):
    """Семантическая стратегия разбиения"""

    def __init__(self):
        self.model = SentenceTransformer(BERT_MODEL_NAME)
        self.similarity_threshold = SIMILARITY_THRESHOLD

    def calculate_death_score(self, left_max, right_max, minimum):
        """
        score = 0.5 * (левый_л_макс + правый_л_макс - 2 * л_мин)
        """
        return 0.5 * (left_max + right_max - 2 * minimum)

    def find_local_maxima(self, series, window=2):
        """Находит локальные максимумы в ряду"""
        maxima = []
        for i in range(window, len(series) - window):
            is_maximum = True
            for j in range(1, window + 1):
                if series[i] < series[i - j] or series[i] < series[i + j]:
                    is_maximum = False
                    break
            if is_maximum:
                maxima.append((i, series[i]))
        return maxima

    def split(self, text):
        """Разбивает текст с анализом схожести между соседними абзацами"""
        raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        if len(raw_paragraphs) <= 1:
            return self.split_large_paragraphs([text])

        similarity_series = self.build_similarity_series(raw_paragraphs)
        merge_decisions = self.analyze_similarity_breaks(similarity_series)
        processed_paragraphs = self.merge_paragraphs_by_decisions(
            raw_paragraphs, merge_decisions
        )
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

            prev_embeddings = embeddings[: len(prev_elements)]
            curr_embeddings = embeddings[len(prev_elements) :]

            max_similarity = 0
            for curr_emb in curr_embeddings:
                for prev_emb in prev_embeddings:
                    similarity = np.dot(curr_emb, prev_emb)
                    max_similarity = max(max_similarity, similarity)

            similarities.append(max_similarity)

        return similarities

    def analyze_similarity_breaks(self, similarity_series):
        """Анализирует провалы в ряду схожестей через Death Score"""
        if len(similarity_series) < 5:
            return [True] * len(similarity_series)

        merge_decisions = []

        local_maxima = self.find_local_maxima(similarity_series, window=1)
        maxima_indices = {idx for idx, _ in local_maxima}

        for i in range(len(similarity_series)):
            current_similarity = similarity_series[i]

            if i in maxima_indices:
                merge_decisions.append(True)
                continue

            left_max = None
            right_max = None

            for j in range(i - 1, -1, -1):
                if j in maxima_indices:
                    left_max = similarity_series[j]
                    break

            for j in range(i + 1, len(similarity_series)):
                if j in maxima_indices:
                    right_max = similarity_series[j]
                    break

            if left_max is not None and right_max is not None:
                death_score = self.calculate_death_score(
                    left_max, right_max, current_similarity
                )
                should_merge = death_score < 0.2
            else:
                should_merge = current_similarity > self.similarity_threshold

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
        sentences = [s.strip() + "." for s in paragraph.split(".") if s.strip()]
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
            fragment_elements = elements[start_idx : split_idx + 1]
            if fragment_elements:
                fragment = " ".join(fragment_elements)
                fragments.append(fragment)
            start_idx = split_idx + 1

        if start_idx < len(elements):
            fragment = " ".join(elements[start_idx:])
            fragments.append(fragment)

        return fragments if fragments else [paragraph]

    def find_semantic_breaks(self, similarities):
        """Находит семантические провалы в ряду схожестей через Death Score"""
        if len(similarities) < 5:
            return []

        breaks = []
        local_maxima = self.find_local_maxima(similarities, window=1)
        maxima_indices = {idx for idx, _ in local_maxima}

        for i in range(2, len(similarities) - 2):
            if i in maxima_indices:
                continue

            left_max = None
            right_max = None

            for j in range(i - 1, -1, -1):
                if j in maxima_indices:
                    left_max = similarities[j]
                    break

            for j in range(i + 1, len(similarities)):
                if j in maxima_indices:
                    right_max = similarities[j]
                    break

            if left_max is not None and right_max is not None:
                death_score = self.calculate_death_score(
                    left_max, right_max, similarities[i]
                )
                if death_score > 0.2 and similarities[i] < self.similarity_threshold:
                    breaks.append(i)

        return breaks
