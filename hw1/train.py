import sys
import csv 
import math
import random
import numpy as np

data = []    
#一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()


x = []
y = []

# i: 月份, j: 某個月中的data base, t: 污染物, s: offset
for i in range(12):
    # 一個月取連續10小時的data可以有471筆(24 * 20 - 9)
    for j in range(471):
        x.append([])
        # 總共有18種污染物
        for t in range(18):
            # 只挑編號2,4,5,8,9,10的污染物  
            if t in [2, 4, 5, 8, 9, 10]:
                # 連續9小時
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s])
                # 平方項
                '''
                if t == 9:
                    for s in range(9):
                        x[471*i+j].append(data[t][480*i+j+s] ** 2)
                '''
        # 第10小時當作答案 
        y.append(data[9][480*i+j+9])

x = np.array(x)
y = np.array(y)

#x = np.concatenate((x,x**2), axis=1)
# 增加平方項

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
w = np.zeros(len(x[0]))       # initial weight vector
lr = 10                         # learning rate
iter = 700000                      # iteration


x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(iter):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# save model
np.save('model.npy',w)
# read model
w = np.load('model.npy')

test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        #for i in range(2,11):
        #    test_x[n_row//18].append(float(r[i]) )
    elif n_row % 18 in [2, 4, 5, 8, 9, 10]:
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
        # 平方項 
        '''
        if n_row % 18 == 9:
            for i in range(2,11):
                if r[i] !="NR":
                    test_x[n_row//18].append(float(r[i]) ** 2)
                else:
                    test_x[n_row//18].append(0)
        '''
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# test_x = np.concatenate((test_x,test_x**2), axis=1)
# 增加平方項

test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
# 增加bias項  


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

