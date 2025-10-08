import os

def read_all_files():
    """функция для почтения всех файлов из папки data"""
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
