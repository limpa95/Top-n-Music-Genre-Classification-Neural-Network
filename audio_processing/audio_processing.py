import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

# some of the code is from librosa documentaiton examples


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


def plot_mel_spectrogram_on_screen(y, sr):
    """"""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='viridis')
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

    # Save the figure to a file
    plt.savefig('test_output/mel_spectrogram.png',
                dpi=300, bbox_inches='tight')

    # plt.show()

    # for testing, remove later
    plt.close()


def select_audio_files():

    print("Please enter the path to a data directory with the song files for that genre.")
    print("The folder structure should be a folder with subfolders named after each genre.")
    print("Inside each genre folder should be the audio files of that genre.")
    data_folder_path = input("Path: ")
    print(f"You entered: {data_folder_path}")

    # genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    genres = []

    for genre_fodler in os.listdir(data_folder_path):
        genres.append(genre_fodler)

    genres.sort()

    return genres, data_folder_path


def calculate_dataset_mel_spectrographs(genres, data_folder_path):
    """"""
    # get songs in genere folder and add them to a list
    for genre in genres:
        print(f"Processing {genre}")
        genre_dir = f'{data_folder_path}/{genre}'
        file_list = []
        for file_name in os.listdir(genre_dir):
            if file_name.endswith('.wav'):
                file_list.append(file_name)
        file_list.sort()

        # go thorugh list of songs and create mel spectroraph
        for audio_name in file_list:
            file_path = os.path.join(genre_dir, audio_name)
            print(
                f"Calculating mel spectrograph for {genre_dir}/{audio_name}")
            y, sr = get_audio_timeseries_array_and_samplerate(file_path)
            plot_mel_spectrogram_on_screen(y, sr)

            # mel_spectrograph = extract_mel_spectrogram(file_path)

        input("Are you ready for the next genre?")
        break


if __name__ == '__main__':

    # while True:

    #     y, sr = enter_single_file()

    #     # plot_timeseries_waveform(y, sr)
    #     plot_mel_spectrogram_on_screen(y, sr)

    # path = "GTZAN_Dataset/Data/genres_original/blues/"
    # files_and_dirs = os.listdir(path)
    # print(files_and_dirs)

    genres, data_folder_path = select_audio_files()
    calculate_dataset_mel_spectrographs(genres, data_folder_path)


# windows
# GTZAN_Dataset\Data\genres_original\blues\blues.00000.wav
# linux
# GTZAN_Dataset/Data/genres_original/blues/blues.00000.wav

# GTZAN_Dataset/Data/genres_original
