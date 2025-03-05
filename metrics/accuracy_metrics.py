import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def display_accuracy(files_list):
    """Loads AI model and formats user submitted spectrogram to produce prediction results.
    Then uses matplotlib to display accuracy metrics in Top-n format."""

    img_size = (128, 128)

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)

    model_path = os.path.join(parent_dir, 'pre_model', 'genre_classification_model.h5')

    new_model = tf.keras.models.load_model(model_path)

    # Grab img path, convert, and normalize spectrogram.

    prediction_avg_list = []

    for file in files_list:
        img_path = os.path.join(parent_dir, 'audio_processing', 'single_output', 'png', file)
        img = Image.open(img_path).convert('RGB').resize(img_size)
        np_array = np.array(img) / 255.0
        np_array = (np.expand_dims(np_array, 0))

        # Make prediction.
        prediction = new_model.predict(np_array)
        prediction_list = prediction[0]

        prediction_avg_list.append(prediction_list)

    prediction_np_averages = np.mean(prediction_avg_list, axis=0)
    prediction_averages = prediction_np_averages.tolist()

    # Pair probabilities with genre labels.
    tuples_list = list(zip(genres, prediction_averages))
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

    predicted_genre = sorted_tuples_list[0][0]

    return predicted_genre

if __name__ == '__main__':
    display_accuracy()
