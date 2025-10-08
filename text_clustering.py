import os
import shutil
from collections import defaultdict

def clean_folder(folder_path):
    """очистка папки со старыми файлами"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def read_all_files():
    """функция для прочтения всех файлов из папки data"""
    texts = []
    filenames = []

    if not os.path.exists('data'):
        return texts, filenames

    for file in os.listdir('data'):
        if file.endswith('.txt'):
            with open(f'data/{file}', 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(content)
                filenames.append(file)

    return texts, filenames

def split_document_themes(text):
    themes_keywords = {
        'финансы': [
            'оплата', 'цена', 'стоимость', 'расчеты', 'платеж', 'взнос',
            'процент', 'рубль', 'доллар', 'евро', 'валюта', 'налог',
            'бюджет', 'финанс', 'сумма', 'денежн', 'вклад', 'депозит'
        ],
        'юриспруденция': [
            'ответственность', 'обязательства', 'штраф', 'неустойка',
            'расторжение', 'договор', 'юридический', 'законодательство',
            'права', 'обязанности', 'споры', 'претензия', 'суд', 'закон'
        ],
        'условия_соглашения': [
            'условия', 'срок', 'порядок', 'правила', 'требования',
            'обязательно', 'должен', 'соглашение', 'предмет договора'
        ],
        'операции_процессы': [
            'поставка', 'доставка', 'транспорт', 'отгрузка', 'приемка',
            'операции', 'процесс', 'выполнение', 'оказание услуг'
        ],
        'техническое': [
            'оборудование', 'технический', 'характеристики', 'гарантия',
            'качество', 'спецификация', 'параметры', 'модель', 'стандарт'
        ],
        'административное': [
            'реквизиты', 'стороны', 'подпись', 'дата', 'номер',
            'адрес', 'контакты', 'паспорт', 'документ'
        ]
    }


    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 30]
    theme_fragments = defaultdict(list)

    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        theme_scores = {}

        for theme, keywords in themes_keywords.items():
            score = sum(1 for keyword in keywords if keyword in paragraph_lower)
            if score > 0:
                theme_scores[theme] = score

        if theme_scores:
            best_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
            theme_fragments[best_theme].append(paragraph)
        else:
            theme_fragments['прочее'].append(paragraph)

    return theme_fragments


def save_fragments(theme_fragments, filename):
    base_name = filename.replace('.txt', '')
    for theme, fragments in theme_fragments.items():
        if fragments:
            with open(f'result_data/{base_name}_{theme}.txt', 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(fragments))

def main():
    clean_folder('result_data')
    texts, filenames = read_all_files()

    for text, filename in zip(texts, filenames):
        fragments = split_document_themes(text)
        save_fragments(fragments, filename)

if __name__ == "__main__":
    main()