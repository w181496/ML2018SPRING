from gensim.models.word2vec import Word2Vec
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, Bidirectional, GRU, Activation
from keras.layers.recurrent import LSTM
from gensim.corpora.dictionary import Dictionary
#from sklearn.cross_validation import train_test_split
import numpy as np
import sys
import os
import csv

cnt = 0

x_test = []

with open(sys.argv[1], 'r') as f:
    for line in f:
        if cnt != 0:
           lines = line.strip().split(',')
           e = []
           tmp = lines[1].split(' ')
           for i in tmp:
              e.append(i)
           x_test.append(e)
        cnt += 1


texts = []
for i in x_test:
    texts.append(i)

# load word2vec model
fname = "word2vec.model"
model = Word2Vec.load(fname)

# Dict
gensim_dict = Dictionary()
gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 詞語的索引，從1開始
w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 詞向量

index_dict = w2indx
word_vectors = w2vec


def text_to_index_array(p_new_dic, p_sen):  # text轉index
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])  # word轉index
            except:
                new_sen.append(0)  # 字典中沒有的word轉0
        new_sentences.append(new_sen)

    return np.array(new_sentences)

x_test = text_to_index_array(index_dict, x_test)
x_test = pad_sequences(x_test, maxlen=140)

# load model
#model = load_model('backup/model1/best.h5')
model = load_model('model.h5')

y_test = model.predict(x_test)

print(y_test.shape)
print(y_test)
print(y_test[0][0])

with open(sys.argv[2], 'w+') as out:
    s = csv.writer(out, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(y_test)):
        if y_test[i][0] > 0.5:
            s.writerow([str(i), str(1)])
        else:
            s.writerow([str(i), str(0)])
