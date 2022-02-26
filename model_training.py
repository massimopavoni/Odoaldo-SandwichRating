import enum
import os
from pickle import dump as pickle_dump, load as pickle_load
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
from tensorflow.python.keras.models import load_model as keras_load_model
from tensorflow_addons import metrics as tfa_metrics


# Dataset enum
class Dataset(enum.Enum):
    train = 0
    validation = 1
    test = 2


# Model class
class OdoaldoSandwichRating:
    def __init__(self, train_batch_size=32, epochs=10):
        self.X = None
        self.y = None
        self.model = None
        self.history = None
        self.train_batch_size = train_batch_size
        self.epochs = epochs

    def __show_rating_distribution(self, values, message):
        # Show rating distribution in dataset
        unique, counts = np.unique(values, return_counts=True)
        plt.cla()
        plt.bar([idx for idx in range(len(unique))], counts)
        plt.title(f"Rating distribution {message}")
        plt.xticks([idx + 0.5 for idx in range(len(unique))], unique, rotation=35, ha='right', size=10)
        plt.show()

    def __oversample_dataset(self, X_in, y_in):
        # Show rating distribution in dataset before oversampling
        self.__show_rating_distribution(y_in, "before oversampling dataset")

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
        self.X = np.concatenate((X_in, X_oversampled))
        self.y = np.concatenate((y_in, y_oversampled))

    def __split_dataset(self):
        # Show rating distribution in dataset before splitting
        self.__show_rating_distribution(self.y, "before splitting dataset")

        # Split dataset
        self.X = np.split(self.X, [int(len(self.X) * 0.8), int(len(self.X) * 0.9)])
        self.y = np.split(self.y, [int(len(self.y) * 0.8), int(len(self.y) * 0.9)])

        # Show rating distribution in dataset
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            unique, counts = np.unique(self.y[i], return_counts=True)
            axs[i].bar([idx for idx in range(len(unique))], counts)
            axs[i].set_xticks([idx + 0.5 for idx in range(len(unique))])
            axs[i].set_xticklabels(unique, rotation=35, ha='right', size=10)
            axs[i].set_title(f"Rating distribution for {Dataset(i).name} set ({len(self.y[i])})")
        fig.tight_layout()
        plt.show()

    def prepare_dataset(self):
        if not os.path.exists(os.path.join('dataset', 'X.pickle')) or not os.path.exists(
                os.path.join('dataset', 'y.pickle')):
            # Deserialize data from pickle format
            with open(os.path.join('dataset', 'X_in.pickle'), 'rb') as pickle_in:
                X_in = pickle_load(pickle_in)
            with open(os.path.join('dataset', 'y_in.pickle'), 'rb') as pickle_in:
                y_in = pickle_load(pickle_in)

            # Normalize data (color values)
            X_in = X_in / 255.0

            self.__oversample_dataset(X_in, y_in)

            # Shuffle dataset
            indices = list(range(len(self.X)))
            random_shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]

            # Serialize data into pickle format
            with open(os.path.join('dataset', 'X.pickle'), 'wb') as pickle_out:
                pickle_dump(self.X, pickle_out)
            with open(os.path.join('dataset', 'y.pickle'), 'wb') as pickle_out:
                pickle_dump(self.y, pickle_out)

            self.__split_dataset()
        else:
            # Deserialize data from pickle format
            with open(os.path.join('dataset', 'X.pickle'), 'rb') as pickle_in:
                self.X = pickle_load(pickle_in)
            with open(os.path.join('dataset', 'y.pickle'), 'rb') as pickle_in:
                self.y = pickle_load(pickle_in)

            self.__split_dataset()

    def build_model(self):
        if not os.path.exists('odoaldo_sandwich_rating_model'):
            # Define model architecture
            self.model = Sequential()

            # First convolutional layer
            self.model.add(Conv2D(32, (3, 3), input_shape=self.X[Dataset.train.value].shape[1:]))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D())

            # Second convolutional layer
            self.model.add(Conv2D(64, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D())

            # Third convolutional layer
            self.model.add(Conv2D(128, (3, 3)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D())

            # Flatten layer
            self.model.add(Flatten())
            # Hidden fully connected layer
            self.model.add(Dense(128))
            self.model.add(Activation('relu'))
            # Output layer
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))

            self.model.summary()

            # Compile model
            self.model.compile(loss=tf_losses.MeanSquaredError(), optimizer=tf_optimizers.nadam_v2.Nadam(),
                               metrics=[tf_metrics.MeanAbsoluteError(), tfa_metrics.RSquare(y_shape=(1,))])
            return True
        else:
            # Load saved model
            self.model = keras_load_model('odoaldo_sandwich_rating_model', compile=False)
            # Recompile loaded module
            self.model.compile(loss=tf_losses.MeanSquaredError(), optimizer=tf_optimizers.nadam_v2.Nadam(),
                               metrics=[tf_metrics.MeanAbsoluteError(), tfa_metrics.RSquare(y_shape=(1,))])
            return False

    def train_model(self):
        # Fit model
        self.history = self.model.fit(self.X[Dataset.train.value], self.y[Dataset.train.value],
                                      batch_size=self.train_batch_size, epochs=self.epochs,
                                      validation_data=(
                                          self.X[Dataset.validation.value], self.y[Dataset.validation.value]))
        # Save trained model
        self.model.save('odoaldo_sandwich_rating_model')

    def evaluate_model(self):
        # Evaluate model on test set
        print("Evaluation on test data")
        self.model.evaluate(self.X[Dataset.test.value], self.y[Dataset.test.value],
                            batch_size=self.train_batch_size * 4)

    def plot_model_stats_history(self):
        # Plot model stats history
        labels = self.history.history.keys()
        fig, axs = plt.subplots(2, len(labels) // 2, figsize=(5 * len(labels) // 2, 12))
        for i, key in enumerate(list(labels)):
            row = i // (len(labels) // 2)
            col = i % (len(labels) // 2)
            axs[row, col].plot(range(1, self.epochs + 1), self.history.history[key])
            axs[row, col].set_title(key)
        fig.tight_layout(pad=4.0)
        plt.show()

    def show_random_samples(self, number_of_samples=10):
        # Show some sample evaluations
        for i in random_choices(range(len(self.X[Dataset.test.value])), k=number_of_samples):
            plt.imshow(self.X[Dataset.test.value][i])
            plt.title(f"Original label: {self.y[Dataset.test.value][i]} - "
                      f"model prediction: {self.model.predict(np.array([self.X[Dataset.test.value][i]]))[0]}")
            plt.show()


def main():
    # Use if for whatever reason you cannot train on GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    odoaldo_sandwich_rating = OdoaldoSandwichRating(epochs=16)
    odoaldo_sandwich_rating.prepare_dataset()
    if odoaldo_sandwich_rating.build_model():
        odoaldo_sandwich_rating.train_model()
        odoaldo_sandwich_rating.plot_model_stats_history()
    odoaldo_sandwich_rating.evaluate_model()
    odoaldo_sandwich_rating.show_random_samples()


if __name__ == '__main__':
    main()
