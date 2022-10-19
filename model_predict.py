import os
from glob import glob1
from pickle import dump as pickle_dump, load as pickle_load
from random import sample as random_sample

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras import losses as tf_losses, metrics as tf_metrics, optimizers as tf_optimizers
from tensorflow.python.keras.models import load_model as keras_load_model

IMG_SIZE = 128

model = keras_load_model('odoaldo_sandwich_rating_model', compile=False)
model.summary()
model.compile(loss=tf_losses.MeanSquaredError(), optimizer=tf_optimizers.nadam_v2.Nadam(),
              metrics=[tf_metrics.MeanAbsoluteError()])

if not os.path.exists(os.path.join('unknown_data', 'X.pickle')) or not os.path.exists(
        os.path.join('unknown_data', 'y.pickle')):
    X = []
    files_count = len(glob1('unknown_data', '*.jpg'))
    for i in range(files_count):
        image = cv2.imread(os.path.join('unknown_data', f'{i}.jpg'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        X.append(image_array)

    with open(os.path.join('unknown_data', 'labels.txt')) as labels:
        y = []
        for label in labels.readlines():
            y.append(round(int(label) / 10 + 0.1, 1))

    X = np.array(X).reshape((-1, IMG_SIZE, IMG_SIZE, 3))
    X = X / 255.0
    y = np.array(y)

    with open(os.path.join('unknown_data', 'X.pickle'), 'wb') as pickle_out:
        pickle_dump(X, pickle_out)
    with open(os.path.join('unknown_data', 'y.pickle'), 'wb') as pickle_out:
        pickle_dump(y, pickle_out)
else:
    with open(os.path.join('unknown_data', 'X.pickle'), 'rb') as pickle_in:
        X = pickle_load(pickle_in)
    with open(os.path.join('unknown_data', 'y.pickle'), 'rb') as pickle_in:
        y = pickle_load(pickle_in)

# Evaluate model on unknown set
print("Evaluation on unknown data")
model.evaluate(X, y, batch_size=len(y))

for i in random_sample(range(len(X)), k=8):
    plt.imshow(X[i])
    plt.title(f"Original label: {y[i]} - "
              f"model prediction: {model.predict(X[i].reshape((-1, IMG_SIZE, IMG_SIZE, 3)))[0]}")
    plt.show()
