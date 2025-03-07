import os
import io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from flask import Flask, request, send_file, jsonify
import tensorflow as tf
from PIL import Image
from flask_cors import CORS

import json
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)
CORS(app)
plt.switch_backend('agg')

CLIENT_ID = '53ce03981b224f6390e23c33329b67aa'
CLIENT_SECRET = 'cd9881b784e74c94abbec986d0438b06'
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))


# List of genres (used for labeling the prediction results)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load the trained model from model.h5 in the root directory
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../pre_model/genre_classification_model.h5"
)
model = tf.keras.models.load_model(MODEL_PATH)


def get_tracks_by_genre(genre, limit=10):
    """Fetch tracks from Spotify based on genre"""
    try:
        results = sp.search(q=f'genre:{genre}', type='track', limit=limit)
        tracks = results['tracks']['items']
        return [
            {"name": track["name"], "artists": [artist["name"] for artist in track["artists"]]}
            for track in tracks
        ]
    except Exception as e:
        print(f"Error fetching tracks from Spotify: {e}")
        return []


# Function to create and save spectrograms for 5-second chunks
def create_spectrogram(audio_path, chunk_duration=10):
    audio_path.seek(0)
    y, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    if total_duration < chunk_duration:
        raise ValueError("Audio file is shorter than required 5 seconds.")

    # Calculate the start time for the middle 5 seconds
    middle_start = (total_duration - chunk_duration) / 2.0
    start_sample = int(middle_start * sr)
    end_sample = int((middle_start + chunk_duration) * sr)
    y_chunk = y[start_sample:end_sample]
    # create spectrograms for each chunk
    D = librosa.stft(y_chunk)
    # use librosa.decompose.hpss to create different spectrograms
    H, P = librosa.decompose.hpss(D)

    plt.figure(figsize=(10, 12))

    # create 1st spectrogram as Full power spectrogram
    plt.subplot(4, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Full power spectrogram')

    # create 2nd spectrogram as Harmonic power spectrogram
    plt.subplot(4, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(H, ref=np.max),
                             y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Harmonic power spectrogram')

    # create 3rd spectrogram as Percussive power spectrogram
    plt.subplot(4, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(P, ref=np.max),
                             y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Percussive power spectrogram')

    # create 4th spectrogram as Sub-band onset strength
    plt.subplot(4, 1, 4)
    onset_subbands = librosa.onset.onset_strength_multi(
        y=y_chunk, sr=sr, channels=[0, 32, 64, 96, 128]
    )
    librosa.display.specshow(onset_subbands, x_axis='time')
    plt.ylabel('Sub-bands')
    plt.title('Sub-band onset strength')

    plt.tight_layout()

    # Save the figure
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()


def predict_genre_from_spectrogram(spectrogram_bytes):
    """
    Opens the spectrogram image from bytes buffer, resizes it to 128x128,
    normalizes it, and runs the model prediction.
    Returns the prediction probabilities as a list.
    """
    img = Image.open(io.BytesIO(spectrogram_bytes)).convert('RGB')
    img = img.resize((128, 128))
    np_img = np.array(img) / 255.0
    np_img = np.expand_dims(np_img, axis=0)
    prediction = model.predict(np_img)
    return prediction[0]


def create_prediction_chart(prediction_list):
    """
    Creates a horizontal bar chart from the prediction_list
    (one probability per genre).
    Returns the bar chart as PNG bytes.
    """
    tuples_list = list(zip(GENRES, prediction_list))
    sorted_tuples = sorted(tuples_list, key=lambda x: x[1])
    genres_sorted = [t[0] for t in sorted_tuples]
    accuracies = [t[1] * 100 for t in sorted_tuples]

    fig = plt.figure()
    plt.barh(genres_sorted, accuracies, 0.5, color='blue')
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Genre")
    plt.title("Music Genre Classification")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle frontend POST requests.
    Expects form-data with 'file' containing a .wav file.
    Returns a PNG image (bar chart) with prediction results.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Create the spectrogram (middle 5 seconds) from the uploaded file
        spectrogram_bytes = create_spectrogram(file)
        # Get prediction from the model
        prediction_list = predict_genre_from_spectrogram(spectrogram_bytes)
        predicted_genre = GENRES[np.argmax(prediction_list)]
        # Save the predicted genre to prediction.json
        recommended_tracks = get_tracks_by_genre(predicted_genre)
        # Create a matplotlib chart
        chart_bytes = create_prediction_chart(prediction_list)
        response_data = {
            "genre": predicted_genre,
            "chart": io.BytesIO(chart_bytes).read().hex(),
            "playlist": recommended_tracks
        }
        print("Response data:", response_data)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run Flask on port 5000
    app.run(host='0.0.0.0', port=5000)
