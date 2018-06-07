from gensim.models.word2vec import Word2Vec
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, GRU, Activation, Bidirectional
from keras.layers.recurrent import LSTM
from gensim.corpora.dictionary import Dictionary
from sklearn.cross_validation import train_test_split
from keras.callbacks import ModelCheckpoint
import sys
import numpy as np

X = []
Y = []
with open(sys.argv[1],'r') as f:
    for line in f:
        lines = line.strip().split(' +++$+++ ')

        # 句子分詞
        tmp = lines[1].split(' ')
        e = []
        for i in tmp:
            e.append(i)
        X.append(e)

        Y.append(int(lines[0]))

# size 詞向量維度
# min_count 詞出現頻率，比這少就不被訓練
# workers 線程數
# sg sg=1表示採用skip-gram, sg=0表示採用cbow
model = Word2Vec(X, size=100, min_count=3, workers=4, sg=1)

print ( model.wv['hello'] )

# Dict
gensim_dict = Dictionary()
gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 詞語的索引，從1開始
w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 詞向量

index_dict = w2indx
word_vectors = w2vec

# save word2vec model
fname = "word2vec.model"
model.save(fname)
#model = Word2Vec.load(fname)
sim = model.wv.most_similar('hello')

for each in sim:
    print(each[0], each[1])

# 算餘弦相似度
#model.similarity('hello','good')

def text_to_index_array(p_new_dic, p_sen):
    new_sentences = []
    for sen in p_sen:
        new_sen = []
        for word in sen:
            try:
                new_sen.append(p_new_dic[word])
            except:
                new_sen.append(0)
        new_sentences.append(new_sen)

    return np.array(new_sentences)



# argument
BATCH_SIZE = 64
EPOCH = 10
vocab_dim = 100
maxlen = 140
input_length = 140

# data preprocessing
n_symbols = len(index_dict) + 1  # 索引數字的個數
embedding_weights = np.zeros((n_symbols, vocab_dim))  # 建立一個n_symbols * 100的0矩陣
for w, index in index_dict.items():  # 從索引1開始，用詞向量填充矩陣
    embedding_weights[index, :] = word_vectors[w]  # 詞向量矩陣,第一行是0向量

X = text_to_index_array(index_dict, X)
X = pad_sequences(X, maxlen=maxlen)
Y = np.array(Y)

# split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

# Build Model
model = Sequential()
# using pretrain weights
model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True, weights=[embedding_weights], input_length=input_length))
#model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(LSTM(output_dim=512, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(LSTM(output_dim=512, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
checkpoint = ModelCheckpoint(filepath="backup/best.h5",verbose=1,save_best_only=True,monitor='val_acc',mode='max')
model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH, validation_data=(x_test, y_test), callbacks = [checkpoint])

# svae model
model.save('model.h5')
