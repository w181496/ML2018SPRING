import sys
import csv
import math
import random
import numpy as np

def predict(test, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inverse)
    x = np.transpose(test)
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1) / N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    return y

def sigmoid(x):
    s = np.divide(1.0, (1.0 + np.exp(-x)))
    return np.clip(s, 1e-8, 1-(1e-8))

x = []
y = []

for i in range(32561):
    x.append([])

# 讀train_X，123個feature
with open('train_X', 'r') as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        if row_cnt == 0:
            print(len(r)) # 123
        else:
            for i in range(123):
                x[row_cnt - 1].append(float(r[i]))
        row_cnt += 1
           
# 讀train_Y
with open('train_Y', 'r') as text:
    row = csv.reader(text, delimiter=',')
    for r in row:
        y.append(r[0])

x = np.array(x)
y = np.array(y).astype('int32')

# Normalization
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x = (x - x_mean) / x_std

train_data_size = x.shape[0]
dim = x.shape[1]

# calculate mu1 and mu2
mu1 = np.zeros((dim,))
mu2 = np.zeros((dim,))

tx = x.sum(axis=0)
cnt1 = 0
cnt2 = 0

for i in range(train_data_size):
    if y[i] == 1:
        mu1 += x[i]
        cnt1 += 1
    else:
        mu2 += x[i]
        cnt2 += 1
mu1 /= cnt1
mu2 /= cnt2

# calculate sigma1 and sigma2
sigma1 = np.zeros((dim,dim))
sigma2 = np.zeros((dim,dim))

for i in range(train_data_size):
    if y[i] == 1:
        sigma1 += np.dot(np.transpose([x[i] - mu1]), ([x[i] - mu1]))
    else :
        sigma2 += np.dot(np.transpose([x[i] - mu2]), ([x[i] - mu2]))

sigma1 /= cnt1
sigma2 /= cnt2

shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2

# 存model
param = {'mu1':mu1, 'mu2':mu2, 'N1':cnt1, 'N2':cnt2, 'shared_sigma':shared_sigma}
np.save('g_model.npy', param)

# 讀model 
param = np.load('g_model.npy')
mu1 = param.item().get('mu1')
mu2 = param.item().get('mu2')
cnt1 = param.item().get('N1')
cnt2 = param.item().get('N2')
shared_sigma = param.item().get('shared_sigma')

test = []
for i in range(16281):
    test.append([])

# 讀test
with open('test_X', 'r') as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        if row_cnt == 0:
            print("attr len", len(r)) # 123
        else:
            for i in range(123):
                test[row_cnt - 1].append(float(r[i]))
        row_cnt += 1
 
test = np.array(test)

# Normalization
test = (test - x_mean) / x_std

z = predict(test, mu1, mu2, shared_sigma, cnt1, cnt2)

print(z)

with open('output.csv', 'w+') as out:
    s = csv.writer(out, delimiter=',', lineterminator='\n')
    s.writerow(["id", "label"])
    for i in range(len(z)):
        if z[i] > 0.5:
            s.writerow([str(i+1), '1'])
        else:
            s.writerow([str(i+1), '0'])

