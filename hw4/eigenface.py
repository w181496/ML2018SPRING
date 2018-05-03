import numpy as np
from skimage import io
import sys
import os

dir_path = sys.argv[1]
target_path = sys.argv[2]

all_file = os.listdir(dir_path)

face_arr = []
for i in all_file:
    img = io.imread(os.path.join(dir_path, i))
    face_arr.append(img.flatten())

face_arr = np.array(face_arr)
face_mean = face_arr.mean(axis=0)
X = face_arr - face_mean

M = np.dot(X, X.T)
e, EV = np.linalg.eig(M)
print(EV.shape)

tmp = np.dot(X.T,EV).T
S = np.nan_to_num(np.sqrt(e))
U = [tmp[i]/S[i] for i in range(12)]

# ==== plot eigenface ====
#img = -U[1].reshape(600,600,3)
#img -= np.min(img)
#img /= np.max(img)
#img = (img*255).astype(np.uint8)

#io.imshow(img)
# =========================

U = np.array(U)


target = io.imread(os.path.join(dir_path, target_path)).flatten()
target = target - face_mean
#print(V.shape)
print(target.shape)
res = np.dot(target, U[:4].T)
print(res)
a = res[0] * U[0]
for i in range(3):
  a += res[i + 1] * U[i + 1]

a = a + face_mean
a = a.reshape(600,600,3)
a -= np.min(a)
a /= np.max(a)
a = (a*255).astype(np.uint8)

#io.imshow(a)
io.imsave("reconstruction.jpg", a)
