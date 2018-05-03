import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import load_model
from keras.utils import np_utils
import sys
import csv

x_test = []

def normalization(x):
    x = (x - x.mean()) / x.std()
    return x

with open("test.csv") as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        if row_cnt != 0:
            tmp = r[1].strip().split(' ')
            for t in tmp:
                x_test.append((float)(t))
        row_cnt += 1

x_test = np.array(x_test).reshape(-1, 48, 48, 1)
x_test = normalization(x_test)

model = load_model("0.67651.h5")
model2 = load_model("0.68041.h5")
model3 = load_model("0.67344.h5")

p = model.predict(x_test)
p2 = model2.predict(x_test)
p3 = model3.predict(x_test)

pred_y = []

for i in range(len(p)):
    a = np.array(p[i])
    b = np.array(p2[i])
    c = np.array(p3[i])
    pred_y.append(np.argmax((a+b+c)/3))


with open("output.csv", 'w+') as out:
    s= csv.writer(out, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(pred_y)):
        s.writerow([str(i), str(pred_y[i])])
