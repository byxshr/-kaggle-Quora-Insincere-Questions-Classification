# -*- coding:UTF-8 -*-
import os
import gc
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import *
from sklearn import metrics
from keras.callbacks import *
from keras.optimizers import *
from unidecode import unidecode
from keras.initializers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras import initializers, regularizers, constraints, optimizers, layers

tqdm.pandas()

embed_size = 300  # 词向量维度
max_features = None  # 特征词数量
maxlen = 72  # 语句序列词数
puncts = [  # 特殊字符列表
    ",", ".", "'", ":", ")", "(", "-", "!", "?", "|", ";", "\"", "$", "&", "/",
    "[", "]", ">", "%", "=", "#", "*", "+", "\\", "•", "~", "@", "£", "·", "_",
    "{", "}", "©", "^", "®", "`", "<", "→", "°", "€", "™", "›", "♥", "←", "×", "§",
    "″", "′", "Â", "█", "½", "à", "…", "“", "★", "”", "–", "●", "â", "►", "−",
    "¢", "²", "¬", "░", "¶", "↑", "±", "¿", "▾", "═", "¦", "║", "―", "¥", "▓", "—",
    "‹", "─", "▒", "：", "¼", "⊕", "▼", "▪", "†", "■", "’", "▀", "¨", "▄", "♫",
    "☆", "é", "¯", "♦", "¤", "▲", "è", "¸", "¾", "Ã", "⋅", "‘", "∞", "∙", "）",
    "↓", "、", "│", "（", "»", "，", "♪", "╩", "╚", "³", "・", "╦", "╣", "╔", "╗",
    "▬", "❤", "ï", "Ø", "¹", "≤", "‡", "√",
]


def clean_text(x):  # 在特殊字符的两侧添加空格
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f" {punct} ")
    return x


""" 读取数据集 """
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
print("训练集形状：{}".format(train.shape))
print("测试集形状：{}".format(test.shape))
""" 将评论中的字母全部改为小写 """
train["question_text"] = train["question_text"].str.lower()
test["question_text"] = test["question_text"].str.lower()
""" 给评论中的特殊字符两边添加空格 """
train["question_text"] = train["question_text"].apply(lambda x: clean_text(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_text(x))
""" 填写空值 """
X = train["question_text"].fillna("_na_").values
X_test = test["question_text"].fillna("_na_").values
""" 设置词标记 """
tokenizer = Tokenizer(num_words=max_features, filters="")
tokenizer.fit_on_texts(list(X))
X = tokenizer.texts_to_sequences(X)
X_test = tokenizer.texts_to_sequences(X_test)
""" 统一语句长度 """
X = pad_sequences(X, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
""" 获取标签值 """
Y = train["target"].values
sub = test[["qid"]]
del train, test  # 删除训练集与测试集
gc.collect()  # 检查内存
word_index = tokenizer.word_index
max_features = len(word_index) + 1  # 初始化max_features不设置初始值，在Tokenizer方法中自己决定word_index数量大小


def load_glove(word_index):  # 导入glove的词向量模型
    EMBEDDING_FILE = "input/embeddings/glove.840B.300d/glove.840B.300d.txt"

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_fasttext(word_index):  # 导入wiki的词向量模型
    EMBEDDING_FILE = "input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100 and o.split(" ")[0] in word_index)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_para(word_index):  # 导入para的词向量模型
    EMBEDDING_FILE = "input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors="ignore") if
                            len(o) > 100 and o.split(" ")[0] in word_index)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


""" 三种embedding """
embedding_matrix_1 = load_glove(word_index)
# embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)
embedding_matrix = np.mean((embedding_matrix_1, embedding_matrix_3), axis=0)
# embedding_matrix = np.hstack((embedding_matrix_1,embedding_matrix_2,embedding_matrix_3))
del embedding_matrix_1, embedding_matrix_3
gc.collect()  # 检查内存
np.shape(embedding_matrix)


def squash(x, axis=-1):  # 归一化
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation="default", **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == "default":
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name="capsule_kernel",
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer="glorot_uniform",
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name="capsule_kernel",
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer="glorot_uniform",
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs,
                                          (0, 2, 1, 3))  # 最后形状[None,num_capsule,input_num_capsule,dim_capsule]
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # 形状[None,num_capsule,input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # 形状[None,input_num_capsule,num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def capsule():
    K.clear_session()
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.2)(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True, kernel_initializer=glorot_normal(seed=12300),
                               recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)
    x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
    x = Flatten()(x)
    x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12300))(x)
    x = Dropout(0.12)(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(), )
    return model


def f1_smart(y_true, y_pred):
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    return 2 * fs[res_idx], (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2


kfold = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
bestscore = []
y_test = np.zeros((X_test.shape[0],))
for i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):
    X_train, X_val, Y_train, Y_val = X[train_index], X[valid_index], Y[train_index], Y[valid_index]
    filepath = "weights_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=2, save_best_only=True, mode="min")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=1, min_lr=0.0001, verbose=2)
    earlystopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=2, verbose=2, mode="auto")
    callbacks = [checkpoint, reduce_lr]
    model = capsule()
    if i == 0: print(model.summary())
    model.fit(X_train, Y_train, batch_size=512, epochs=6, validation_data=(X_val, Y_val), verbose=2,
              callbacks=callbacks, )
    model.load_weights(filepath)
    y_pred = model.predict([X_val], batch_size=1024, verbose=2)
    y_test += np.squeeze(model.predict([X_test], batch_size=1024, verbose=2)) / 5
    f1, threshold = f1_smart(np.squeeze(Y_val), np.squeeze(y_pred))
    print("Optimal F1: {:.4f} at threshold: {:.4f}".format(f1, threshold))
    bestscore.append(threshold)

y_test = y_test.reshape((-1, 1))
pred_test_y = (y_test > np.mean(bestscore)).astype(int)
sub = pd.read_csv('input/sample_submission.csv')
sub["prediction"] = pred_test_y
sub.to_csv("submission.csv", index=False)