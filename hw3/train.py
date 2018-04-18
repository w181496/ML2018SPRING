import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, LeakyReLU, PReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import History ,ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import sys
import csv

x = []
y = []
x_test = []

def normalization(x):
    x = (x - x.mean()) / x.std()
    return x

with open(sys.argv[1]) as text:
    row = csv.reader(text, delimiter=',')
    row_cnt = 0
    for r in row:
        if row_cnt != 0:
            tmp = r[1].strip().split(' ')
            for t in tmp:
                x.append((float)(t))
            y.append((float)(r[0]))
        row_cnt += 1
    print(row_cnt)

x = np.array(x, dtype=float).reshape(-1, 48, 48, 1)
y = np.array(y, dtype=float)
y = np_utils.to_categorical(y, num_classes=7)

x = normalization(x)
#x_test = normalization(x_test)

model = Sequential()

# Layer 1
model.add(Conv2D(filters=64, kernel_size=(4,4), input_shape=(48, 48, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(filters=128, kernel_size=(4,4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Layer 3
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Layer 4
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# Layer 5
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

check  = ModelCheckpoint("drive/ML/batchnormal-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)


model.fit(x, y, batch_size = 192, epochs = 1200, validation_split=0.1,callbacks=[check])

model.save("model.h5")

