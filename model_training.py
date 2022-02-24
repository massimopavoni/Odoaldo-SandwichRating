import os
from pickle import load as pickle_load
from random import choices as random_choices, shuffle as random_shuffle

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras import losses as tf_losses, metrics as tf_metrics, optimizers as tf_optimizers
from tensorflow.python.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D
)
from tensorflow.python.keras.models import Sequential
from tensorflow_addons import metrics as tfa_metrics

# Deserialize data from pickle format
with open(os.path.join('dataset', 'X.pickle'), 'rb') as pickle_in:
    X_in = pickle_load(pickle_in)
with open(os.path.join('dataset', 'y.pickle'), 'rb') as pickle_in:
    y_in = pickle_load(pickle_in)

# Normalize data (color values)
X_in = X_in / 255.0

# Oversample data
unique, unique_index, unique_count = np.unique(y_in, return_inverse=True, return_counts=True)
count = np.max(unique_count)
X_oversampled = np.empty((count * len(unique) - len(X_in),) + X_in.shape[1:], X_in.dtype)
y_oversampled = np.empty((count * len(unique) - len(y_in),) + y_in.shape[1:], y_in.dtype)
slices = np.concatenate(([0], np.cumsum(count - unique_count)))
for i in range(len(unique)):
    indices = np.random.choice(np.where(unique_index == i)[0], count - unique_count[i])
    X_oversampled[slices[i]:slices[i + 1]] = X_in[indices]
    y_oversampled[slices[i]:slices[i + 1]] = y_in[indices]
X_oversampled = np.concatenate((X_in, X_oversampled))
y_oversampled = np.concatenate((y_in, y_oversampled))

# Shuffle dataset
indices = list(range(len(X_oversampled)))
random_shuffle(indices)
X = X_oversampled[indices]
y = y_oversampled[indices]

# Split dataset
X = [X_train, X_validation, X_test] = np.split(X, [int(len(X) * 0.8), int(len(X) * 0.9)])
y = [y_train, y_validation, y_test] = np.split(y, [int(len(y) * 0.8), int(len(y) * 0.9)])

# Print dataset sizes
print(f"Train set size: {len(X_train)}"
      f"\nValidation set size: {len(X_validation)}"
      f"\nTest set size: {len(X_test)}")

# Show rating distribution in dataset
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
labels = ['train', 'validation', 'test']
for i in range(3):
    unique, counts = np.unique(y[i], return_counts=True)
    axs[i].bar([idx for idx in range(len(unique))], counts)
    axs[i].set_xticks([idx + 0.5 for idx in range(len(unique))])
    axs[i].set_xticklabels(unique, rotation=35, ha='right', size=10)
    axs[i].set_title(f"Rating distribution for {labels[i]} set")
fig.tight_layout()
plt.show()

# !!!!!!!!!!!!!!!! Temporary no gpu acceleration training !!!!!!!!!!!!!!!!
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define model topology
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Compile model
model.compile(loss=tf_losses.MeanSquaredError(), optimizer=tf_optimizers.nadam_v2.Nadam(),
              metrics=[tf_metrics.MeanAbsoluteError(), tfa_metrics.RSquare(y_shape=(1,))])

# Fit model
history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_validation, y_validation))

# Evaluate model on test set
print("Evaluation on test data")
results = model.evaluate(X_test, y_test, batch_size=128)

# Show some sample evaluations
for i in random_choices(range(len(X_test)), k=10):
    plt.imshow(X_test[i])
    plt.show()
    print(model.predict(np.array([X_test[i]]))[0])
