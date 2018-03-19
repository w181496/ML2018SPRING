import sys
import csv 
import math
import random
import numpy as np

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
    ans[i].append((int)(a+0.5))

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
