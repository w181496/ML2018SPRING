import numpy as np
from skimage import io

face_arr = []
for i in range(415):
    img = io.imread('./Aberdeen/'+str(i)+'.jpg')
    face_arr.append(img)

face_arr = np.array(face_arr)

avg = face_arr.mean(axis=0).astype(np.uint8)

io.imsave("avg.jpg",avg)
