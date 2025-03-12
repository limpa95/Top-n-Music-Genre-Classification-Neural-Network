
# Top-n-Music-Genre-Classification-Neural-Network

## Program Link (No installation needed):

 Website: http://13.59.74.139:5173/


## About The Project

This project is an intuitive music genre identification web application designed to classify music genres by analyzing created spectrograms using a **Convolutional Neural Network (CNN**).  The CNN model was trained on the [**GTZAN**](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset, which comprises WAV audio files categorized into 10 genres. 

## How it works:

This application is divided up through two separate programs: a command-line interface program and a web application.  Both programs offer a different way of interacting with the CNN model and return genre predictions along with a recommendation playlist to the user.

[**CNN model**](https://www.geeksforgeeks.org/introduction-convolution-neural-network/)

![alt text](https://media.geeksforgeeks.org/wp-content/uploads/20231218174301/max.png)

In the CLI program, a user can place music files within a designated folder.  The application processes these files, and triggers the CNN model to classify their genres.  Instructions are prompted to the user, and they receive immediate feedback through a bar chart depicting the top predicted genres ranked by accuracy in descending order, along with a list of similar music recommendations from Spotify.
The Web Application allows users to upload a .wav file directly through a front-end interface.  After a user uploads the audio file, the application processes it on the server-side, and predicts the music genres.  The server returns a top-n genre list sorted by confidence values and a list of Spotify playlist that align with the identified genres.  The front-end then takes these values and displays them to the user.

## Preprocessing:

While developing the program, it became evident that preprocessing the music was critical in increasing the accuracy of genre classification.  Initially, using Librosa to generate basic spectrograms-a visual representation of the spectrum of frequencies of the audio signal as it varies with time-resulted in only 50-55% accuracy in genre prediction.  After adjusting the extracted layers and shortening the music file, we were able to increase the accuracy to around 70%.   
Ultimately, we decided to adopt a more streamlined and effective approach:

Once the file is uploaded, the middle 10 second chunk is segmented and used as a representative portion of the song to reduce intro or outro contamination.  Once segmented, the audio is converted into multiple spectrograms:
1. **Full Power Spectrogram**
- The complete frequency range and energy of the audio
2. **Harmonic Power Spectrogram**
- The harmonic elements of the audio to increase tonal features for genre discrimination
3. **Percussive Power Spectrogram**
- The percussive elements for rhythmic attributes
4. **Sub-band onset strength spectrogram**
- The transient events in multiple frequency bands, helping discriminate instrumental and rhythmic patterns.

These four spectrograms are saved as one pre-processed file and collectively feed into the created CNN model for music genre classification.
