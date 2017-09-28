import pickle
import os
from glob import glob


def get_vocab_size_fasttext(file_path):
    with open(file_path, "rb") as file:
        vocab = pickle.load(file)
        return len(vocab)


def get_vocab_size_polyglot(file_path):
    with open(file_path, "rb") as file:
        vocab = pickle.load(file, encoding="latin1")
        return len(vocab[0])



if __name__ == '__main__':
    embedding_folder_fasttext = "../../../word_embeddings/fasttext/"
    embedding_folder_polyglot = "../../../word_embeddings/polyglot/"
    list_vocab_file_fasttext = glob(os.path.join(embedding_folder_fasttext, "*", "vocab_word_embeddings_300.p"))
    list_vocab_file_polyglot = glob(os.path.join(embedding_folder_polyglot, "*", "polyglot-*.pkl"))

    for file in list_vocab_file_fasttext:
        size = get_vocab_size_fasttext(file)
        print(file, size)

    for file in list_vocab_file_polyglot:
        try:
            size = get_vocab_size_polyglot(file)
            print(file, size)
        except:
            print("Error", file)
            continue