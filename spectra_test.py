# Jun Seo

# Description: This script demonstrates how to prepare the data
# to train a neural network model to classify music genres using the 
# GTZAN dataset. create_spectrogram() loads the audio files, divides each song into 5sec 
# seperate file with 4 different spectrograms for each file, 
# and saves them in a format(png) suitable for training a neural network.
# Spectrograms are a visual representation of the spectrum of frequencies 
# in a sound or other signal as they vary with time. 
# 4 different spectrograms(Full, Harmonic, Percussive power and sub band 
# onset strength) are created for each 5sec audio file.
# Once spectrograms are created, they are saved in a folder named 
# "spectrograms" in the same directory as the audio files.
# spectrogram source https://librosa.org/doc/latest/generated/librosa.decompose.hpss.html
# sub-band onset strength source http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/generated/librosa.onset.onset_strength_multi.html#librosa.onset.onset_strength_multi
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os


# Define the path to the audio data
data_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/genres_original'
output_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/spectrograms'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)


# Function to create and save spectrograms for 5-second chunks
def create_spectrogram(audio_path, output_folder, chunk_duration=5):
    y, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(total_duration // chunk_duration)

    for i in range(num_chunks):
        start_sample = int(i * chunk_duration * sr)
        end_sample = int((i + 1) * chunk_duration * sr)
        y_chunk = y[start_sample:end_sample]
        # create spectrograms for each chunk
        D = librosa.stft(y_chunk)
        # use librosa.decompose.hpss to create 3 different spectrograms
        H, P = librosa.decompose.hpss(D)

        plt.figure(figsize=(10, 12))

        # create 1st spectrogram as Full power spectrogram
        plt.subplot(4, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Full power spectrogram')

        # create 2nd spectrogram as Harmonic power spectrogram
        plt.subplot(4, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(H, ref=np.max), y_axis='log', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Harmonic power spectrogram')

        # create 3rd spectrogram as Percussive power spectrogram
        plt.subplot(4, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(P, ref=np.max), y_axis='log', sr=sr)
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
        output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_chunk_{i}.png")
        plt.savefig(output_file)
        plt.close()


# Iterate through the genres and create spectrograms for all audio files in each genre
for genre in genres:
    genre_folder = os.path.join(data_path, genre)
    output_genre_folder = os.path.join(output_path, genre)
    os.makedirs(output_genre_folder, exist_ok=True)
    
    for audio_file in os.listdir(genre_folder):
        if audio_file.endswith('.wav'):  # Assuming the audio files are in .wav format
            audio_path = os.path.join(genre_folder, audio_file)
            print(f"Creating spectrograms for {audio_path}")
            create_spectrogram(audio_path, output_genre_folder)
