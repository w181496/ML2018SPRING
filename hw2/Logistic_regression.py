import sys
import csv
import math
import random
import numpy as np

def sigmoid(x):
    s = np.divide(1.0, (1.0 + np.exp(-x)))
    return np.clip(s, 1e-8, 1-(1e-8))

def cross_entrophy(test_x, test_y, w):
    z = np.dot(test_x, w)
    prob = sigmoid(z)
    cross = -(np.dot(test_y, np.log(prob)) + np.dot(1 - test_y, np.log(1 - prob)))
    return cross

def logistic_regression(x, y, lr, iteration):
    ll = 2
    x_t = x.transpose()
    w = np.zeros(len(x[0]))
    s_gra = np.zeros(len(x[0]))
    for i in range(iteration):
        z = np.dot(x, w)
        prob = sigmoid(z)
        gra = -np.dot(x_t, y-prob) + 2 * ll * w
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        w = w - lr * gra/ada
        print(i, cross_entrophy(x, y, w))
    return w


x = []
y = []

for i in range(32561):
    x.append([])

# 讀train_X，123個feature
with open(sys.argv[3], 'r') as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        #print(row_cnt)
        if row_cnt == 0:
            print(len(r)) # 123
        else:
            for i in range(123):
                #if i != 10:
                x[row_cnt - 1].append(float(r[i]))
        row_cnt += 1
           
# 讀train_Y
with open(sys.argv[4], 'r') as text:
    row = csv.reader(text, delimiter=',')
    for r in row:
        y.append(r[0])

x = np.array(x)
y = np.array(y, dtype=float)

# Normalization
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x = (x - x_mean) / x_std

# 合併bias項
x = np.concatenate((np.ones((x.shape[0],1)),x),axis=1)



iteration = 30000
lr = 10

w = logistic_regression(x, y, lr, iteration)

# 存model
print(w)
np.save('model.npy', w)

# 讀model
# w = np.load('model.npy')

test = []
for i in range(16281):
    test.append([])

# 讀test
with open(sys.argv[5], 'r') as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        if row_cnt == 0:
            print("attr len", len(r)) # 123
        else:
            for i in range(123):
                #if i != 10:
                test[row_cnt - 1].append(float(r[i]))
        row_cnt += 1
 
test = np.array(test)

# Normalization
test = (test - x_mean) / x_std

test = np.concatenate((np.ones((test.shape[0],1)),test),axis=1)
res = np.dot(test, w)
z = sigmoid(res)

with open(sys.argv[6], 'w+') as out:
    s= csv.writer(out, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(z)):
        if z[i] > 0.5:
            s.writerow([str(i+1), '1'])
        else:
            s.writerow([str(i+1), '0'])

