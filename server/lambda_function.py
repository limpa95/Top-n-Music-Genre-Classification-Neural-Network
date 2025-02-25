import os
import sys
import boto3
import zipfile
import io
import time
import base64
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------------
# --- Handle Dependencies ------------------------------------------
# ------------------------------------------------------------------
S3_BUCKET = "capstoneosu"
DEPENDENCY_FILE = "python.zip"
DEPENDENCY_PATH = "/tmp/lambda_dependencies"

def download_and_extract_dependencies():
    """Downloads dependencies from S3 and extracts them to /tmp/"""
    if not os.path.exists(DEPENDENCY_PATH):  # Avoid re-downloading on warm starts
        s3 = boto3.client("s3")
        zip_path = "/tmp/dependencies.zip"

        print(f"Downloading dependencies from s3://{S3_BUCKET}/{DEPENDENCY_KEY}")
        s3.download_file(S3_BUCKET, DEPENDENCY_KEY, zip_path)

        print("Extracting dependencies...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DEPENDENCY_PATH)

        print("Dependencies ready.")

    # Add extracted dependencies to Python path
    sys.path.append(DEPENDENCY_PATH)

# ------------------------------------------------------------------
# --- Preprocessing Code (audio_processing.py) ---------------------
# ------------------------------------------------------------------

# tell librosa to use specific backend. Helps with multicore
os.environ['AUDIORAD_BACKEND'] = 'ffmpeg'


def get_audio_timeseries_array_and_samplerate(audio_path):
    """
    Gets the waveform amplitude and the sample rate its sampled at by
    librosa.
    """
    # y is a numpy array of the waveform amplitude
    # sr is the sample rate which defaults to 22050Hz
    y, sr = librosa.load(audio_path)
    return y, sr


def plot_timeseries_waveform(y, sr):
    """
    Displays the time series waveform of an audio file using the amplitude
    and sample rate.
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform of the Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def convert_audio_to_mel_spectrogram(y, sr, show_plot=False, show_axis=False):
    """
    Takes the audio file y and sr to convert it to a mel spectrogram
    shifted to dB.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    if show_axis is True:
        img = librosa.display.specshow(
            S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
    else:
        img = librosa.display.specshow(
            S_dB, sr=sr, fmax=8000, ax=ax)
    if show_plot is True:
        plt.show()
    return fig, S_dB


def save_mel_spectrogram_png(fig, audio_name, genre=""):
    """Saves the mel spectrogram as a png"""

    # strip .wav from audio name
    stripped_name = audio_name.replace(".wav", "")

    if genre != "":
        # some of this function was generated with Copilot
        genre_path = os.path.join('mel_spec_training_png_out', f'{genre}')

        # Create the directory if it does not exist
        if not os.path.exists(genre_path):
            os.makedirs(genre_path)

        # Full path for the file
        file_path = os.path.join(genre_path, f'{stripped_name}.png')

    else:
        single_file_output_directory = os.path.join("single_output", "png")

        if not os.path.exists(single_file_output_directory):
            os.makedirs(single_file_output_directory)

        file_path = os.path.join(
            single_file_output_directory, f"{stripped_name}.png")

    # Save the figure to a file
    print(f"Saving {file_path}")
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_mel_spectrogram_npy(S_dB, audio_name, genre=""):
    """Saves the mel spectrogram as a npy file"""
    # strip .wav from audio name
    stripped_name = audio_name.replace(".wav", "")

    if genre != "":
        # some of this function was generated with Copilot
        genre_path = os.path.join('mel_spec_training_npy_out', f'{genre}')

        # Create the directory if it does not exist
        if not os.path.exists(genre_path):
            os.makedirs(genre_path)

        # Full path for the file
        file_path = os.path.join(genre_path, f'{stripped_name}.npy')

    else:
        single_file_output_directory = os.path.join("single_output", "npy")

        if not os.path.exists(single_file_output_directory):
            os.makedirs(single_file_output_directory)

        file_path = os.path.join(
            single_file_output_directory, f"{stripped_name}.npy")

    # Save the S_dB to a file
    print(f"Saving {file_path}")
    np.save(file_path, S_dB)
    plt.close()


# The rest of the functions are not used in this pipeline.


# --------------------------------------------------------------------
# --- CNN Prediction and Chart Generation Code (accuracy_metric.py) --
# --------------------------------------------------------------------

def display_accuracy(file_name):
    """Loads AI model and formats user submitted spectrogram to produce prediction results.
    Then uses matplotlib to display accuracy metrics in Top-n format."""

    img_size = (128, 128)

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
              'metal', 'pop', 'reggae', 'rock']

    # Instead of using __file__ to compute paths, we adjust for Lambda
    # Here, we assume that the spectrogram is saved as /tmp/<file_name>.png
    img_path = os.path.join('/tmp', f'{file_name}.png')
    img = Image.open(img_path).convert('RGB').resize(img_size)
    np_array = np.array(img) / 255.0
    np_array = np.expand_dims(np_array, 0)

    # Load the model from /tmp/model.h5 as assumed.
    model_path = '/var/task/model.h5'
    new_model = tf.keras.models.load_model(model_path)

    # Make prediction.
    prediction = new_model.predict(np_array)
    prediction_list = prediction[0]

    # Pair probabilities with genre labels.
    tuples_list = list(zip(genres, prediction_list))
    sorted_tuples_list = sorted(tuples_list, key=lambda x: x[1])

    # Display accuracy metrics in top-n format.
    y_labels = [x[0] for x in sorted_tuples_list]
    accuracies = [x[1] * 100 for x in sorted_tuples_list]
    colors = ["blue"] * len(genres)

    plt.figure(figsize=(10, 6))
    plt.barh(y_labels, accuracies, 0.5, color=colors)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Genre")
    plt.title("Music Genre Classification")
    chart_path = os.path.join('/tmp', f'{file_name}_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    return chart_path


# ------------------------------------------------------------------
# --- AWS Lambda Handler -------------------------------------------
# ------------------------------------------------------------------

def lambda_handler(event, context):
    """
    AWS Lambda handler that accepts a music.wav file from the API,
    performs preprocessing steps to generate input spectrograms,
    feeds the spectrogram into the CNN model to generate top-n-list 
    predictions, and returns the chart image to the user.
    """
    # --- Step 0: Grab dependencies ---
    download_and_extract_dependencies()

    # --- Step 1: Retrieve the .wav file from API---
    if event.get("isBase64Encoded", False):
        wav_data = base64.b64decode(event["body"])
    else:
        wav_data = event["body"].encode('utf-8')
    
    # Save the incoming WAV file to /tmp/music.wav
    wav_file_path = '/tmp/music.wav'
    with open(wav_file_path, 'wb') as wav_file:
        wav_file.write(wav_data)

    # --- Step 2: Generate the Spectrogram---
    y, sr = get_audio_timeseries_array_and_samplerate(wav_file_path)
    fig, S_dB = convert_audio_to_mel_spectrogram(y, sr)

    # Save the spectrogram PNG
    # (Normally, save_mel_spectrogram_png saves to "single_output/png", but here we need it in /tmp.)
    spectrogram_path = os.path.join('/tmp', 'music.png')
    print(f"Saving spectrogram to {spectrogram_path}")
    fig.savefig(spectrogram_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Step 3: Generate the Prediction Chart using the CNN model ---
    chart_path = display_accuracy("music")

    # --- Step 4: Return the Prediction Chart Image via the API ---
    with open(chart_path, "rb") as chart_file:
        chart_data = chart_file.read()
    encoded_chart = base64.b64encode(chart_data).decode('utf-8')

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "image/png"},
        "isBase64Encoded": True,
        "body": encoded_chart
    }
