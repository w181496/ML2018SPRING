import numpy as np
import csv
import sys
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster


image = np.load(sys.argv[1]).astype('float32')

image = image/255
#image.reshape(image, (len(image), -1))

pca = PCA(n_components=441, whiten=True)
newData = pca.fit_transform(image)

kmeans_fit = cluster.KMeans(n_clusters = 2).fit(newData)

res = kmeans_fit.labels_

print("Clustering result:")
print(kmeans_fit.labels_)


y = []
with open(sys.argv[2], 'r') as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        if row_cnt != 0:
            a = int(r[1])
            b = int(r[2])
            if res[a] != res[b]:
                y.append(0)
            else:
                y.append(1)
        row_cnt += 1
#print(y[:100])

with open(sys.argv[3], 'w+') as out:
    s= csv.writer(out, delimiter=',', lineterminator='\n')
    s.writerow(["ID", "Ans"])
    for i in range(len(y)):
        s.writerow([str(i), y[i]])

