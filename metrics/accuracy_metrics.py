import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def display_accuracy():
    """Loads AI model and formats user submitted spectrogram to produce prediction results.
    Then uses matplotlib to display accuracy metrics in Top-n format."""

    img_size = (128, 128)

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)

    model_path = os.path.join(parent_dir, 'model', 'genre_classification_cnn.h5')

    new_model = tf.keras.models.load_model(model_path)

    # Grab img path, convert, and normalize spectrogram.

    img_path = os.path.join(parent_dir, 'single_output', 'png', f'blues.00000_part4.png')
    img = Image.open(img_path).convert('RGB').resize(img_size)
    np_array = np.array(img) / 255.0
    np_array = (np.expand_dims(np_array, 0))

    # Make prediction.
    prediction = new_model.predict(np_array)
    prediction_list = prediction[0]

    # Pair probabilities with genre labels.
    tuples_list = list(zip(genres, prediction_list))
    sorted_tuples_list = sorted(tuples_list, key=lambda x: x[1])

    # Display accuracy metrics in top-n format.
    y = [x[0] for x in sorted_tuples_list]
    w = [x[1] * 100 for x in sorted_tuples_list]
    c = ["blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]

    plt.barh(y, w, 0.5, color=c)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Genre")
    plt.title("Music Genre Classification")
    plt.show()


if __name__ == '__main__':
    display_accuracy()
