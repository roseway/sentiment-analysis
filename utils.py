import numpy as np
from string import punctuation
from os import listdir


def load_emb(filename, vocab):
    f = open(filename, 'r', encoding='UTF-8')
    lines = f.readlines()
    f.close()
    vocab_size = len(vocab) + 1
    wordlist = np.zeros((vocab_size, 300))
    embedding = dict()
    for line in lines:
        x = line.split()
        embedding[x[0]] = np.asarray(x[1:], dtype='float32')
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            wordlist[i] = vector
    return wordlist


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='UTF-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def clean_doc(doc, vocab):
    # split into tokens by white space
    doc = doc.replace('<br />', ' ')
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens


def prepare(directory, vocab, num=False):
    documents = list()
    if num:
        i = 1
        for filename in listdir(directory):
            path = directory + '/' + filename
            doc = load_doc(path)
            tokens = clean_doc(doc, vocab)
            documents.append(tokens)
            if i >= num:
                break
            i += 1
    else:
        for filename in listdir(directory):
            path = directory + '/' + filename
            doc = load_doc(path)
            tokens = clean_doc(doc, vocab)
            documents.append(tokens)
    return documents
