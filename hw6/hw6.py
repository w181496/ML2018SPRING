from keras import regularizers
from keras.layers import Input, Embedding, Dense, BatchNormalization, Flatten, dot, add, merge
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import  preprocessing
import pandas as pd
import csv
import numpy as np
import sys

test = pd.read_csv(sys.argv[1], names=["id", "user_id", "movie_id"], header=0)
users = pd.read_csv(sys.argv[4], names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], delimiter='::', header=0)
movies = pd.read_csv(sys.argv[3], names=['movie_id', 'title', 'genres'], delimiter='::', header=0)

model = load_model('model.h5')
pred = model.predict([test.user_id, test.movie_id]).squeeze()
pred = pred * 4 + 1.05
pred = np.clip(pred, 1, 5)
result = [[i+1, pred[i]] for i in range(len(pred))]
df = pd.DataFrame(result, columns = ["TestDataID", "Rating"])
df.to_csv(sys.argv[2], index=False)
