from utils import load_doc
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from os import listdir

min_occur = 2
doc_dir = ['data/Imdb/train/neg', 'data/Imdb/train/pos']
save_vocab = 'data/vocab.txt'


def clean_doc(doc):
    # split into tokens by white space
    doc = doc.replace('<br />', ' ')
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)


def process_docs(directory, vocab):
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path, vocab)


print("Extracting vocabulary")
vocab = Counter()
# add all docs to vocab
for dire in doc_dir:
    process_docs(dire, vocab)
# keep tokens with a min occurrence
tokens = [k for k, c in vocab.items() if c >= min_occur]
# convert lines to a single blob of text
data = '\n'.join(tokens)
file = open(save_vocab, 'w', encoding='UTF-8')
file.write(data)
file.close()
print("Done")
