from abc import ABC, abstractmethod
import re


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