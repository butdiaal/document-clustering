# import argparse
# import os
# import shutil
# from collections import defaultdict
#
#
# def read_all_files():
#     """функция для прочтения всех файлов из папки data"""
#     texts = []
#     filenames = []
#
#     if not os.path.exists('data'):
#         return texts, filenames
#
#     for file in os.listdir('data'):
#         if file.endswith('.txt'):
#             with open(f'data/{file}', 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 texts.append(content)
#                 filenames.append(file)
#
#     return texts, filenames
#
#
#
# def save_fragments(theme_fragments, filename):
#     base_name = filename.replace('.txt', '')
#     for theme, fragments in theme_fragments.items():
#         if fragments:
#             with open(f'result_data/{base_name}_{theme}.txt', 'w', encoding='utf-8') as f:
#                 f.write('\n\n'.join(fragments))
#
# def main():
#     parser = argparse.ArgumentParser(description='Кластеризация документов по тематикам')
#     parser.add_argument('--input', '-i', default='data',
#                         help='Входная директория')
#     parser.add_argument('--output', '-o', default='result_data',
#                         help='Выходная директория')
#
#     args = parser.parse_args()
#
#     os.makedirs(args.output, exist_ok=True)
#
#
# if __name__ == "__main__":
#     main()