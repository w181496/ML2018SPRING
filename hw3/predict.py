import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import load_model
from keras.utils import np_utils
import sys
import csv

x_test = []

def normalization(x):
    #x = (x - x.mean()) / x.std()
    # this is training data mean and std
    x = (x - 129.47433955331468) / 65.02727348443116
    return x

with open(sys.argv[1]) as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        if row_cnt != 0:
            tmp = r[1].strip().split(' ')
            for t in tmp:
                x_test.append((float)(t))
        row_cnt += 1

x_test = np.array(x_test).reshape(-1, 48, 48, 1)
print(x_test.mean, x_test.std)
x_test = normalization(x_test)

# public and private model are the same
model = load_model("model.h5")

p = model.predict(x_test)

pred_y = []
for i in p:
    print(i)
    pred_y.append(np.argmax(i))

with open(sys.argv[2], 'w+') as out:
    s= csv.writer(out, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(pred_y)):
        s.writerow([str(i), str(pred_y[i])])

