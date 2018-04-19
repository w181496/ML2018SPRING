from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
from keras.models import load_model
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
import csv

x_test = []

def normalization(x):
    x = (x - x.mean()) / x.std()
    return x

x = []
y = []
with open("train.csv") as text:
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

model = load_model("model.h5")

class_idx = 3
indices = np.where(y[:, class_idx] == 1.)[0]

# pick some random input from here.
idx = indices[0]

# Lets sanity check the picked image.
plt.rcParams['figure.figsize'] = (18, 6)

plt.imshow(x[idx][..., 0])
plt.show()

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
#layer_idx = utils.find_layer_idx(model, 'preds')
layer_idx = -1

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x[idx])
# Plot with 'jet' colormap to visualize as a heatmap.
plt.imshow(grads, cmap='jet')
plt.show()
