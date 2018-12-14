from utils import *
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

vocab_dir = 'data/vocab.txt'
embedding_dir = 'data/glove.6B.300d.txt'
train_dir = ['data/Imdb/train/neg', 'data/Imdb/train/pos']
test_dir = ['data/Imdb/test/neg', 'data/Imdb/test/pos']
# The number of reviews wanna extract, use False to extract all
num = False

# Get vocab
vocab = load_doc(vocab_dir)
vocab = set(vocab.split())

# Prepare training data
print("Preparing training and test data")
train_docs = list()
for d in train_dir:
    train_docs += prepare(d, vocab, num)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = max([len(s) for s in encoded_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytrain = np.array([0 for _ in range(num if num else 12500)] + [1 for _ in range(num if num else 12500)])
ytrain = to_categorical(ytrain, num_classes=2)

# Prepare test data
test_docs = list()
for d in test_dir:
    test_docs += prepare(d, vocab, num)
encoded_docs = tokenizer.texts_to_sequences(test_docs)
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytest = np.array([0 for _ in range(num if num else 12500)] + [1 for _ in range(num if num else 12500)])
ytest = to_categorical(ytest, num_classes=2)

# Using pre-trained word embedding
print("Loading Word Embedding")
wordlist = load_emb(embedding_dir, tokenizer.word_index)
embedding_layer = Embedding(vocab_size, 300, weights=[wordlist], input_length=max_length, trainable=False)
# Define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.summary()
# Compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit network
model.fit(Xtrain, ytrain, epochs=20, verbose=1)

# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc * 100))
