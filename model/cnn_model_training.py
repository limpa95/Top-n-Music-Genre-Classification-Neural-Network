import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image


# Constants
DATA_DIR = "data/spectrogram"
IMG_SIZE = (128, 128)  # For resizing images to a fixed size
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_FILE = "model/genre_classification_cnn.h5"


def load_data(data_dir, img_size):
    """
    Load the spectrogram data and labels from each directory image.
    Resizes the images and normalizes them for the model.
    """
    spec_data = []
    genre_labels = []
    genres = sorted(os.listdir(data_dir))
    label_map = {genre: i for i, genre in enumerate(genres)}
    for genre in genres:
        genre_dir = os.path.join(data_dir, genre)
        if os.path.isdir(genre_dir):
            for file_name in os.listdir(genre_dir):
                if file_name.endswith(".png"):
                    img_path = os.path.join(genre_dir, file_name)
                    img = Image.open(img_path).convert('RGB').resize(img_size)
                    spec_data.append(np.array(img) / 255.0)
                    genre_labels.append(label_map[genre])
    spec_data = np.array(spec_data)
    genre_labels = np.array(genre_labels)
    return spec_data, genre_labels, genres


def create_cnn(input_shape, num_classes):
    """
    Build a Convolutional Neural Network for spectrogram classification.
    Uses a sequential model to stack layers in linear order.  These are
    the layers used:
    1. Conv2D : Applies convolutional filters to each image and extracts
    features.
    2. MaxPooling2D : Reduces the spatial dimensions.
    3. Dropout : Randomly drops nodes during training time,
    which helps prevent overfitting.
    4. Flatten : Flattens the input, converting it to a 1D array.
    5. Dense : Fully connected layer that outputs the final classification.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=input_shape
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # Flatten and Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128,
            activation='relu'
        ),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(
            num_classes,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(model, training_data, training_labels, validation_data,
                validation_labels, batch_size, epochs):
    """
    Train the CNN model created using create_cnn() and returns
    the training history, including accuracy values.  This also
    stops training if validation loss does not improve.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        training_data, training_labels,
        validation_data=(validation_data, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )
    return history


def evaluate_model(model, testing_data, testing_labels, genres):
    """
    The model evaluates itself after each epoch by printing a
    classification report to the user.  This tests the model
    accuracy on unseen data by comparing the predicted vs
    actual results from the model.
    """
    test_loss, test_accuracy = model.evaluate(
        testing_data,
        testing_labels,
        verbose=0
        )
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    predicted = np.argmax(model.predict(testing_data), axis=1)
    actual = np.argmax(testing_labels, axis=1)
    print(classification_report(actual, predicted, target_names=genres))


def plot_training_history(history):
    """
    Plots the training and validation accuracy over epochs
    using matplotlib.  This is useful for showing the
    user a visualization of the model's training performance.
    """
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()


def save_model(model, path):
    """
    Save the trained model to a file within the model folder.
    """
    model.save(path)
    print(f"Model saved as '{path}'")


def main():
    # Load the dataset
    spec_data, genre_labels, genres = load_data(DATA_DIR, IMG_SIZE)
    print(f"Loaded {len(spec_data)} images across {len(genres)} genres.")

    # Split data into train/test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        spec_data, genre_labels, test_size=0.2, random_state=42,
        stratify=genre_labels
    )

    # Convert labels to categorical
    num_classes = len(genres)
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    # Create the model
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    model = create_cnn(input_shape, num_classes)
    model.summary()

    # Train the model
    history = train_model(
        model,
        train_data,
        train_labels,
        test_data,
        test_labels,
        BATCH_SIZE,
        EPOCHS
    )

    # Evaluate the model
    evaluate_model(model, test_data, test_labels, genres)

    # Plot training history
    plot_training_history(history)

    # Save the model
    save_model(model, MODEL_SAVE_FILE)


if __name__ == "__main__":
    main()
