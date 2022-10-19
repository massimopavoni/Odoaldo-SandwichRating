import enum
import os
from pickle import dump as pickle_dump, load as pickle_load
from random import choices as random_choices, shuffle as random_shuffle
from sys import argv as sys_argv

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras import losses as tf_losses, metrics as tf_metrics, optimizers as tf_optimizers
from tensorflow.python.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model as keras_load_model

# Numpy RNG object
np_rng = np.random.default_rng()


# Dataset enum
class Dataset(enum.Enum):
    train = 0
    validation = 1
    test = 2


def show_rating_distribution(values, message):
    # Show rating distribution in dataset
    unique, counts = np.unique(values, return_counts=True)
    plt.cla()
    plt.bar([idx for idx in range(len(unique))], counts)
    plt.title(f"Rating distribution {message}")
    plt.xticks([idx + 0.5 for idx in range(len(unique))], unique, rotation=35, ha='right', size=10)
    plt.show()


class OdoaldoSandwichRating:
    # Model class

    def __init__(self, train_batch_size=32, epochs=8):
        self.X = None
        self.y = None
        self.model = None
        self.history = None
        self.train_batch_size = train_batch_size
        self.epochs = epochs

    def __oversample_dataset(self, X_in, y_in):
        # Oversample data
        unique, unique_index, unique_count = np.unique(y_in, return_inverse=True, return_counts=True)
        max_count = np.max(unique_count)
        X_oversampled = np.empty((max_count * len(unique) - len(X_in),) + X_in.shape[1:], X_in.dtype)
        y_oversampled = np.empty((max_count * len(unique) - len(y_in),) + y_in.shape[1:], y_in.dtype)
        slices = np.concatenate(([0], np.cumsum(max_count - unique_count)))
        for i in range(len(unique)):
            indices = np_rng.choice(np.where(unique_index == i)[0], max_count - unique_count[i], replace=True)
            X_oversampled[slices[i]:slices[i + 1]] = X_in[indices]
            y_oversampled[slices[i]:slices[i + 1]] = y_in[indices]
        self.X = np.concatenate((X_in, X_oversampled))
        self.y = np.concatenate((y_in, y_oversampled))

    def __undersample_dataset(self, X_in, y_in):
        # Undersample data
        unique, unique_index, unique_count = np.unique(y_in, return_inverse=True, return_counts=True)
        min_count = np.min(unique_count)
        X_undersampled = np.empty((min_count * len(unique),) + X_in.shape[1:], X_in.dtype)
        y_undersampled = np.empty((min_count * len(unique),) + y_in.shape[1:], y_in.dtype)
        for i in range(len(unique)):
            indices = np_rng.choice(np.where(unique_index == i)[0], min_count, replace=False)
            X_undersampled[min_count * i:min_count * (i + 1)] = X_in[indices]
            y_undersampled[min_count * i:min_count * (i + 1)] = y_in[indices]
        self.X = X_undersampled
        self.y = y_undersampled

    def __averagesample_dataset(self, X_in, y_in):
        # Average-sample data
        unique, unique_index, unique_count = np.unique(y_in, return_inverse=True, return_counts=True)
        average_count = int(np.mean(unique_count))
        X_averagesampled = np.empty((average_count * len(unique),) + X_in.shape[1:], X_in.dtype)
        y_averagesampled = np.empty((average_count * len(unique),) + y_in.shape[1:], y_in.dtype)
        for i in range(len(unique)):
            indices = np.where(unique_index == i)[0]
            if unique_count[i] == average_count:
                pass
            elif unique_count[i] > average_count:
                indices = np_rng.choice(indices, average_count, replace=False)
            else:
                additional_indices = np_rng.choice(indices, average_count - unique_count[i], replace=True)
                indices = np.concatenate((indices, additional_indices))
            X_averagesampled[average_count * i:average_count * (i + 1)] = X_in[indices]
            y_averagesampled[average_count * i:average_count * (i + 1)] = y_in[indices]
        self.X = X_averagesampled
        self.y = y_averagesampled

    def __split_dataset(self):
        # Show rating distribution in dataset before splitting
        show_rating_distribution(self.y, "before splitting dataset")

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

            # Show rating distribution in dataset before balancing
            show_rating_distribution(y_in, "before balancing dataset")

            choice = input("Do you want to oversample, undersample or averagesample\n"
                           "(oversampling for lower than average, undersampling for higher than average) the dataset?\n"
                           "(o/u/a, or enter to continue without modifying original dataset) ")
            if choice == 'o':
                self.__oversample_dataset(X_in, y_in)
            elif choice == 'u':
                self.__undersample_dataset(X_in, y_in)
            elif choice == 'a':
                self.__averagesample_dataset(X_in, y_in)
            else:
                self.X = X_in
                self.y = y_in

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
            self.model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                  input_shape=self.X[Dataset.train.value].shape[1:]))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # Second convolutional layer
            self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                  input_shape=self.X[Dataset.train.value].shape[1:]))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # Third convolutional layer
            self.model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1)))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            # Flatten layer
            self.model.add(Flatten())

            # First dense layer
            self.model.add(Dense(512))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))

            # Second dense layer
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))

            # Third dense layer
            self.model.add(Dense(128))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.5))

            # Output layer
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))

            self.model.summary()

            # Compile model
            self.model.compile(loss=tf_losses.MeanSquaredError(), optimizer=tf_optimizers.nadam_v2.Nadam(),
                               metrics=[tf_metrics.MeanAbsoluteError()])
            return True
        else:
            # Load saved model
            self.model = keras_load_model('odoaldo_sandwich_rating_model', compile=False)

            self.model.summary()

            # Recompile loaded module
            self.model.compile(loss=tf_losses.MeanSquaredError(), optimizer=tf_optimizers.nadam_v2.Nadam(),
                               metrics=[tf_metrics.MeanAbsoluteError()])
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

    def show_random_samples(self, number_of_samples=8):
        # Show some sample evaluations
        for i in random_choices(range(len(self.X[Dataset.test.value])), k=number_of_samples):
            plt.imshow(self.X[Dataset.test.value][i])
            plt.title(f"Original label: {self.y[Dataset.test.value][i]} - "
                      f"model prediction: {self.model.predict(np.array([self.X[Dataset.test.value][i]]))[0]}")
            plt.show()


def main():
    # Use if for whatever reason you cannot train on GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    train_batch_size = int(sys_argv[1]) if len(sys_argv) > 1 else 32
    epochs = int(sys_argv[2]) if len(sys_argv) > 2 else 12

    odoaldo_sandwich_rating = OdoaldoSandwichRating(train_batch_size=train_batch_size, epochs=epochs)
    odoaldo_sandwich_rating.prepare_dataset()
    if odoaldo_sandwich_rating.build_model():
        odoaldo_sandwich_rating.train_model()
        odoaldo_sandwich_rating.plot_model_stats_history()
    odoaldo_sandwich_rating.evaluate_model()
    odoaldo_sandwich_rating.show_random_samples()


if __name__ == '__main__':
    main()
