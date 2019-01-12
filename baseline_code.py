# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the i...
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import tensorflow as tf
import keras
from keras import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
K = keras.backend
L = keras.layers

maxlen = 72     #Complete a sentence or reserve to 72 words

def load_and_prec():
    """
    Load datasets and thesaurus
    """
    df_train = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")
    ## fill up the missing values
    train_X = df_train["question_text"].fillna("_##_").values
    test_X = df_test["question_text"].fillna("_##_").values
    ## Tokenize the sentences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    ## Pad the setences
    x_train = pad_sequences(train_X, maxlen=maxlen)
    x_test = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    y_train = df_train['target'].values

    return x_train, x_test, y_train, tokenizer.word_index



def load_glove(word_index, max_features):
    """
    Load word vector
    """
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf8'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def FastText(embedding_matrix):
    """
    Define FaqstTest model
    """
    ipt = L.Input(shape=(maxlen,))
    x = L.Embedding(input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    trainable=False)(ipt)
    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=ipt, outputs=out)

    return model

x_train, x_test, y_train, word_index = load_and_prec()
max_features = len(word_index)+1
embedding_matrix = load_glove(word_index, max_features)

model = FastText(embedding_matrix)
model.compile(
    loss='binary_crossentropy',
    optimizer = 'adam',
)
hist = model.fit(x_train, y_train,
                 batch_size=256,
                 epochs=1,
                 verbose=1,)

##############Submmit########
y_pred = model.predict(x_test)
sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = y_pred > 0.2
sub.to_csv("submission.csv", index=False)
