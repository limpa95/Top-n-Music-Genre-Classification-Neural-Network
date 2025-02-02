import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import multiprocessing
import time


"""
This file is the main file used for converting audio into formates usefull for
training the AI model.
It is also used in main for converting single files into formats that will be
input into the completed AI model for genre classificaiton.
"""


# some of the code is derived from librosa documentaiton examples

# tell librosa to use specific backend. Helps with multicore
os.environ['AUDIORAD_BACKEND'] = 'ffmpeg'


def get_audio_timeseries_array_and_samplerate(audio_path):
    """
    Gets the waveform amplitude and the saple rate its sampled at by
    librosa
    """

    # y is a numpy array of the waveform amplitude
    # sr is the sample rate which defaults to 22050Hz
    y, sr = librosa.load(audio_path)
    return y, sr


def plot_timeseries_waveform(y, sr):
    """
    Displays the itme series waveform of an audio file using the amplitude
    and sample rate
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
    shifted to dB
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


def training_input_file_path(prompt=True):
    """
    Used for collect multiple files in genre folders for the purpose of
    generating training data
    """
    if prompt is True:
        print(
            "Please enter the path to a data directory with the song files \
                for that genre.")
        print("The folder structure should be the data folder with subfolders \
              named after each genre.")
        print("Inside each genre folder should be the audio files of that \
              genre you would like converted.")
        print("Example: 'GTZAN_Dataset/Data/genres_original'")
    data_folder_path = input("Path: ")
    # print(f"You entered: {data_folder_path}")

    # genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', \
    # 'metal', 'pop', 'reggae', 'rock']
    genres = []

    for genre_fodler in os.listdir(data_folder_path):
        genres.append(genre_fodler)

    genres.sort()

    return genres, data_folder_path


def convert_single_file_mel_spectrogam(file):
    """Converts a single file to a mel spectrogram and saves it as a
    png and npy file.
    """
    file_path = os.path.join("single_input", file)
    print(f"Calculating mel spectrogram for {file}")
    y, sr = get_audio_timeseries_array_and_samplerate(file_path)
    fig, S_dB = convert_audio_to_mel_spectrogram(y, sr)
    save_mel_spectrogram_png(fig, file)
    save_mel_spectrogram_npy(S_dB, file)


# def validate_single_file(file_path):
#     """"""
#     return file_path


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


def process_single_audio_file(file, conversion):
    """"""
    conversion(file)


def process_audio_files(all_files, conversion, parallell=False):
    """"""
    start = time.time()
    if parallell is False:
        for file in all_files:
            conversion(file)

    else:
        # Use multiprocessing to process the audio files in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as \
                pool:
            pool.map(conversion, all_files)

    end = time.time()

    print(f"Time: {(end - start)} seconds")


# Function to process each audio file
# The way the args are passed to this funciton were
# influenced by copilot output
def convert_dataset_mel_spectrogram(args):
    """
    Converts multiple files in genre folders into mel spectrograms and saves
    them as png and npy files
    """
    genre, genre_dir, audio_name = args
    file_path = os.path.join(genre_dir, audio_name)
    print(f"Calculating mel spectrogram for {file_path}")
    y, sr = get_audio_timeseries_array_and_samplerate(file_path)
    fig, S_dB = convert_audio_to_mel_spectrogram(y, sr)
    save_mel_spectrogram_png(fig, audio_name, genre)
    save_mel_spectrogram_npy(S_dB, audio_name, genre)


def save_mel_spectrogram_png(fig, audio_name, genre=""):
    """Saves the mel spectrogram as a png"""

    # strip .wav from audio name
    stripped_name = audio_name.replace(".wav", "")

    if genre != "":
        # some of this funciton was generated with Copilot
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
        # some of this funciton was generated with Copilot
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


class EmptyDirectoryError(Exception):
    """Custom exception for empty directories."""
    pass


def check_single_input_directory(EmptyDirectoryError, data_folder_path):
    """
    Checks for and gets the file names if there are any in the directory
    passed in
    """
    try:
        # get list of files in directory
        files_in_directory = os.listdir(data_folder_path)

        # check if directry is empty
        if not files_in_directory:
            raise EmptyDirectoryError(
                f"The directory {data_folder_path} is empty.")

        print(f"Found {files_in_directory}")
        return files_in_directory

    except FileNotFoundError:
        print(f"The directory '{data_folder_path}' does not exist.")
        print(f"Making directory: {data_folder_path}")
        os.mkdir(data_folder_path)
        return

    except EmptyDirectoryError:
        # print(empty)
        return


def remove_converted_audio_file(data_folder_path, audio_file):
    try:
        os.remove(os.path.join(data_folder_path, audio_file))

    except Exception:
        print("An error occurred removing the audio file after converison")


if __name__ == '__main__':

    # -----------------------------------------------------
    # Program for converting multiple genre folders of files.

    try:

        genres, data_folder_path = training_input_file_path()
        all_files = get_audio_files(genres, data_folder_path)
        process_audio_files(
            all_files, convert_dataset_mel_spectrogram, parallell=True)

    except KeyboardInterrupt:
        print("Exiting the program.")

    # -------------------------------------------------------------------


# -----------------------------------------------------------------
# file path examples for converting lots of files at once.

# windows
# GTZAN_Dataset\Data\genres_original

# linux
# GTZAN_Dataset/Data/genres_original


# GTZAN_Dataset/Data/genres_original/blues/blues.00090.wav
# GTZAN_Dataset/Data/genres_original/blues/blues.00091.wav
