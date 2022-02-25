import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras import losses as tf_losses, metrics as tf_metrics, optimizers as tf_optimizers
from tensorflow.python.keras.models import load_model as keras_load_model
from tensorflow_addons import metrics as tfa_metrics

IMG_SIZE = 128

for file in os.listdir('custom_images'):
    image = cv2.imread(os.path.join('custom_images', file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    X = np.array(image_array).reshape((IMG_SIZE, IMG_SIZE, 3))
    X = X / 255.0
    model = keras_load_model('odoaldo_sandwich_rating_model', compile=False)
    model.compile(loss=tf_losses.MeanSquaredError(), optimizer=tf_optimizers.nadam_v2.Nadam(),
                  metrics=[tf_metrics.MeanAbsoluteError(), tfa_metrics.RSquare(y_shape=(1,))])
    plt.imshow(image)
    plt.title(f"Model prediction: {model.predict(np.array([X]))[0]}")
    plt.show()
