import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import multiprocessing
import time

# some of the code is derived from librosa documentaiton examples

# tell librosa to use specific backend. Helps with multicore
os.environ['AUDIORAD_BACKEND'] = 'ffmpeg'


def get_audio_timeseries_array_and_samplerate(audio_path):
    """"""

    # y is a numpy array of the waveform amplitude
    # sr is the sample rate which defaults to 22050Hz
    y, sr = librosa.load(audio_path)
    return y, sr


def plot_timeseries_waveform(y, sr):
    """"""
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform of the Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def convert_audio_to_mel_spectrogram(y, sr, show_plot=False, show_axis=False):
    """"""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    if show_axis is True:
        img = librosa.display.specshow(
            S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)  # cmap='viridis'
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
    else:
        img = librosa.display.specshow(
            S_dB, sr=sr, fmax=8000, ax=ax)

    if show_plot is True:
        plt.show()

    return fig


def input_file_path(prompt=False):
    """"""
    if prompt is True:
        print(
            "Please enter the path to a data directory with the song files for that genre.")
        print("The folder structure should be a folder with subfolders named after each genre.")
        print("Inside each genre folder should be the audio files of that genre.")
    data_folder_path = input("Path: ")
    # print(f"You entered: {data_folder_path}")

    # genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    genres = []

    for genre_fodler in os.listdir(data_folder_path):
        genres.append(genre_fodler)

    genres.sort()

    return genres, data_folder_path


def get_audio_files(genres, data_folder_path):
    """"""
    all_files = []
    # Get all audio files in all genres
    for genre in genres:
        print(f"Getting {genre}")
        genre_dir = f'{data_folder_path}/{genre}'
        for file_name in os.listdir(genre_dir):
            if file_name.endswith('.wav'):
                all_files.append((genre, genre_dir, file_name))

    return all_files


def process_audio_file(all_files, conversion, parallell=False):
    """"""
    start = time.time()
    if parallell is False:
        for file in all_files:
            conversion(file)

    else:
        # Use multiprocessing to process the audio files in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(conversion, all_files)

    end = time.time()

    print(f"Time: {(end - start)} seconds")


# Function to process each audio file
# The way the args are passed to this funciton were influenced by copilot output
def convert_dataset_mel_spectrogram(args):
    genre, genre_dir, audio_name = args
    file_path = os.path.join(genre_dir, audio_name)
    print(f"Calculating mel spectrogram for {file_path}")
    y, sr = get_audio_timeseries_array_and_samplerate(file_path)
    fig = convert_audio_to_mel_spectrogram(y, sr)
    save_mel_spectrogram(fig, genre, audio_name)


def save_mel_spectrogram(fig, genre, audio_name):
    """"""
    # some of this funciton was generated with Copilot
    folder_path = f'mel_spectrogram/{genre}'

    # Create the directory if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # strip .wav from audio name
    stripped_name = audio_name.replace(".wav", "")

    # Full path for the file
    file_path = os.path.join(folder_path, f'{stripped_name}.png')

    # Save the figure to a file
    fig.savefig(file_path, dpi=300, bbox_inches='tight')

    plt.close()


if __name__ == '__main__':

    genres, data_folder_path = input_file_path()
    all_files = get_audio_files(genres, data_folder_path)
    process_audio_file(
        all_files, convert_dataset_mel_spectrogram, parallell=False)


# file path examples

# windows
# GTZAN_Dataset\Data\genres_original

# linux
# GTZAN_Dataset/Data/genres_original
