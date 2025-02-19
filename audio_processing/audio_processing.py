import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import multiprocessing
import time


"""
This file is the main file used for converting audio into formats useful for
training the AI model.
It is also used in main for converting single files into formats that will be
input into the completed AI model for genre classification.
"""


# some of the code is derived from librosa documentation examples

# tell librosa to use specific backend. Helps with multi-core
os.environ['AUDIORAD_BACKEND'] = 'ffmpeg'


def get_audio_timeseries_array_and_sample_rate(audio_path):
    """
    Gets the waveform amplitude and the sample rate its sampled at by
    librosa
    """

    # y is a numpy array of the waveform amplitude
    # sr is the sample rate which defaults to 22050Hz
    y, sr = librosa.load(audio_path)
    return y, sr


def break_audio_into_sections(y, sr, length=5):
    """
    Breaks the audio samples up into discrete sections based on length
    passed in.
    """
    samples = len(y)
    length_seconds = len(y)/sr
    audio_chunks = []
    print(f"Sample rate: {sr}")
    print(f"Number of samples: {samples}")
    print(f"Audio length (s): {int(length_seconds)}")

    if length_seconds < 5 or length < 5:
        print("Audio must be at least 5 seconds long")
        audio_chunks.append(y)
        return y, sr

    elif length_seconds > length:

        # make it a integer to cut off the last few seconds of the audio that
        # wont be full length
        number_of_chunks = int((samples / sr) / length)
        print(f"Number of chunks: {number_of_chunks}")
        samples_per_chunk = length * sr
        print(f"Samples per chunk: {samples_per_chunk}")

        for chunk in range(number_of_chunks):
            if chunk == 0:
                start = 0
            end = samples_per_chunk * (chunk + 1)
            print(f"Chunk {chunk} Start: {start}")
            print(f"Chunk {chunk} End: {end}")
            audio_chunks.append(y[start:end - 1])
            start = end

        return audio_chunks, sr

    else:
        print("Requested audio length is longer then audio.")
        audio_chunks.append(y)
        return y, sr


def get_middle_of_audio(y, sr, length=30):
    """
    Gets a middle portion of the audio samples based on the length passed in.
    """
    samples = len(y)
    length_seconds = len(y)/sr
    audio_chunks = []
    print(f"Sample rate: {sr}")
    print(f"Number of samples: {samples}")
    print(f"Audio length (s): {int(length_seconds)}")

    if length_seconds > length:
        # print(length_seconds - length)
        samples_to_trim = round((length_seconds - 30) * sr)
        print(f"Number of samples to trim: {samples_to_trim}")
        print(f"Trimming {round(samples_to_trim/2)} samples from both ends")
        trimed_y = y[int(samples_to_trim/2):-int(samples_to_trim/2)]

        audio_chunks.append(trimed_y)
        return trimed_y, sr

    else:
        print("Requested audio length is longer then audio.")
        audio_chunks.append(y)
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


def convert_audio_to_multi_spectrogram(y, sr):
    """
    This function utilizes code from the pre_model/spectra_test.py file to 
    keep the same data shape for the AI model. 
    """
    D = librosa.stft(y)
    # use librosa.decompose.hpss to create 3 different spectrograms
    H, P = librosa.decompose.hpss(D)

    plt.figure(figsize=(10, 12))

    # create 1st spectrogram as Full power spectrogram
    plt.subplot(4, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                             y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Full power spectrogram')

    # create 2nd spectrogram as Harmonic power spectrogram
    plt.subplot(4, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(H), ref=np.max),
                             y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Harmonic power spectrogram')

    # create 3rd spectrogram as Percussive power spectrogram
    plt.subplot(4, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(P), ref=np.max),
                             y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Percussive power spectrogram')

    # create 4th spectrogram as Sub-band onset strength
    plt.subplot(4, 1, 4)
    onset_subbands = librosa.onset.onset_strength_multi(
        y=y, sr=sr, channels=[0, 32, 64, 96, 128]
    )
    librosa.display.specshow(onset_subbands, x_axis='time')
    plt.ylabel('Sub-bands')
    plt.title('Sub-band onset strength')

    plt.tight_layout()

    return plt


def training_input_file_path(prompt=True):
    """
    Used for collect multiple files in genre folders for the purpose of
    generating training data
    """
    if prompt is True:
        print(
            "Please enter the path to a data directory with the song files \
                for that genre.")
        print("The folder structure should be the data folder with subfolder \
              named after each genre.")
        print("Inside each genre folder should be the audio files of that \
              genre you would like converted.")
        print("Example: 'GTZAN_Dataset/Data/genres_original'")
    data_folder_path = input("Path: ")
    # print(f"You entered: {data_folder_path}")

    # genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', \
    # 'metal', 'pop', 'reggae', 'rock']
    genres = []

    for genre_folder in os.listdir(data_folder_path):
        genres.append(genre_folder)

    genres.sort()

    return genres, data_folder_path


def convert_single_file_mel_spectrogram(file):
    """Converts a single file to a mel spectrogram and saves it as a
    png and npy file.
    """
    start = time.time()
    file_path = os.path.join("single_input", file)
    print(f"Calculating mel spectrogram for {file}")
    y, sr = get_audio_timeseries_array_and_sample_rate(file_path)
    y_middle, sr = get_middle_of_audio(y, sr, 30)
    audio_chunks, sr = break_audio_into_sections(y_middle, sr, 5)

    # process each audio chunk of an audio file.
    for chunk in range(len(audio_chunks)):
        fig, S_dB = convert_audio_to_mel_spectrogram(audio_chunks[chunk], sr)
        partial_file_name = file + f"_part{chunk}"
        save_mel_spectrogram_png(fig, partial_file_name)
        save_mel_spectrogram_npy(S_dB, partial_file_name)

    end = time.time()

    print(f"Time to convert audio file: {(end - start)} seconds")


def convert_single_file_multi_spectrogram(file):
    """Converts a single file to a mel spectrogram and saves it as a
    png and npy file.
    """
    start = time.time()
    file_path = os.path.join("single_input", file)
    print(f"Calculating multi spectrogram for {file}")
    y, sr = get_audio_timeseries_array_and_sample_rate(file_path)
    y_middle, sr = get_middle_of_audio(y, sr, 30)
    audio_chunks, sr = break_audio_into_sections(y_middle, sr, 5)

    # process each audio chunk of an audio file.
    for chunk in range(len(audio_chunks)):
        plt = convert_audio_to_multi_spectrogram(audio_chunks[chunk], sr)
        partial_file_name = file + f"_part{chunk}"
        save_multi_spectrogram_png(plt, partial_file_name)

    end = time.time()

    print(f"Time to convert audio file: {(end - start)} seconds")


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


def process_audio_files(all_files, conversion, parallel=False):
    """"""
    start = time.time()
    if parallel is False:
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
# The way the args are passed to this function were
# influenced by copilot output
def convert_dataset_mel_spectrogram(args):
    """
    Converts multiple files in genre folders into mel spectrograms and saves
    them as png and npy files
    """
    start = time.time()

    genre, genre_dir, audio_name = args
    file_path = os.path.join(genre_dir, audio_name)
    print(f"Calculating mel spectrogram for {file_path}")
    y, sr = get_audio_timeseries_array_and_sample_rate(file_path)
    # audio_chunks, sr = get_middle_of_audio(y, sr, 30)
    audio_chunks, sr = break_audio_into_sections(y, sr, 5)

    # process each audio chunk of an audio file.
    for chunk in range(len(audio_chunks)):
        fig, S_dB = convert_audio_to_mel_spectrogram(audio_chunks[chunk], sr)
        partial_audio_name = audio_name + f"_part{chunk}"
        save_mel_spectrogram_png(fig, partial_audio_name, genre)
        save_mel_spectrogram_npy(S_dB, partial_audio_name, genre)

    end = time.time()
    print(f"Time to convert audio file: {(end - start)} seconds")


def convert_dataset_multi_spectrogram(args):
    """
    Converts multiple files in genre folders into multi spectrograms and saves
    them as png files
    """
    start = time.time()

    genre, genre_dir, audio_name = args
    file_path = os.path.join(genre_dir, audio_name)
    print(f"Calculating multi spectrogram for {file_path}")
    y, sr = get_audio_timeseries_array_and_sample_rate(file_path)
    # audio_chunks, sr = get_middle_of_audio(y, sr, 30)
    audio_chunks, sr = break_audio_into_sections(y, sr, 5)

    # process each audio chunk of an audio file.
    for chunk in range(len(audio_chunks)):
        plt = convert_audio_to_multi_spectrogram(audio_chunks[chunk], sr)
        partial_audio_name = audio_name + f"_part{chunk}"
        save_multi_spectrogram_png(plt, partial_audio_name, genre)

    end = time.time()
    print(f"Time to convert audio file: {(end - start)} seconds")


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


def save_multi_spectrogram_png(plt, audio_name, genre=""):
    """Saves the mel spectrogram as a png"""

    # strip .wav from audio name
    stripped_name = audio_name.replace(".wav", "")

    if genre != "":
        # some of this function was generated with Copilot
        genre_path = os.path.join('multi_spec_training_png_out', f'{genre}')

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
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
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

        # check if directory is empty
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
        print("An error occurred removing the audio file after conversion")


if __name__ == '__main__':

    # -----------------------------------------------------
    # Program for converting multiple genre folders of files.

    try:

        genres, data_folder_path = training_input_file_path()
        all_files = get_audio_files(genres, data_folder_path)
        # process_audio_files(
        #     all_files, convert_dataset_mel_spectrogram, parallel=True)
        process_audio_files(
            all_files, convert_dataset_multi_spectrogram, parallel=True)

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
