# Jun Seo

# Description: This program trains a neural network model to classify music genres using the GTZAN dataset.
# spectra_test.py is used to create spectrograms for each 5sec audio file and save them in a folder named 
# "spectrograms" in the same directory as the audio files.
# load_data() loads the spectrograms(.png) and labels from the specified data path.
# The model is created with multiple convolutional layers, max pooling layers, batch normalization layers, and dropout layers.
# The model is trained with early stopping and learning rate reduction callbacks.
# The model is trained using the spectrograms as input features and the genre labels as output labels.
# The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
# The model also uses data augmentation to improve the model's performance.
# The model is saved in the .h5 format.
# The predict_genre() function takes the path to an audio file as input and returns the predicted genre.
# The predict_tempo() function(extended goal) takes the path to an audio file as input and returns the predicted tempo in BPM.
# The tempo is estimated using the librosa library by calculating the onset strength and beat track.
# The tempo is then categorized into one of the following tempo markings: Grave, Largo, Adagio, Andante, Moderato, Allegro, Vivace, Presto.
# The tempo markings are based on the tempo ranges defined by the Italian musical terms.
# The predicted genre and tempo are printed to the console.
#source: https://keras.io/examples/audio/stft/

import numpy as np
import librosa
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(data_path, genres, img_size=(128, 128)):
    '''
    spectra_test.py is used to create spectrograms for each 5sec audio file and save them 
    in a folder named "spectrograms" in the same directory as the audio files.
    load_data() loads the spectrograms(.png) and labels from the specified data path.
    '''
    X = []
    y = []  # labels
    for genre in genres:
        genre_folder = os.path.join(data_path, genre)
        for img_file in os.listdir(genre_folder):
            if img_file.endswith('.png'):
                img_path = os.path.join(genre_folder, img_file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(img_size)
                img_array = np.array(img)  # extract first dimension of the shape using NumPy array
                X.append(img_array)
                y.append(genre)
    return np.array(X), np.array(y)


def create_model(input_shape, num_genres):
    '''
    The model is created with multiple convolutional layers, max pooling layers, batch normalization layers, and dropout layers.
    '''
    model = Sequential()
    # Apply convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # Max pooling layer to reduce spatial dimensions
    model.add(MaxPooling2D((2, 2)))
    # Batch normalization to stabilize learning
    model.add(BatchNormalization())
    # Dropout for regularization
    model.add(Dropout(0.25))

    # Apply more convolutional layers
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Apply more convolutional layers
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    # Flatten the output for the dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # Apply output layer
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(optimizer=Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model


def model_train(model, X_train, y_train, X_val, y_val):
    '''
    The model is trained with early stopping and learning rate reduction callbacks.
    The model is trained using the spectrograms as input features and the genre labels as output labels.
    The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
    The model also uses data augmentation to improve the model's performance.
    '''
    # Early stopping callback if training loss does not improve for 10 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Reduce learning rate on plateau callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Train the model with early stopping and learning rate reduction
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_val, y_val),
                        epochs=100,
                        shuffle=True,
                        verbose=2,
                        callbacks=[early_stopping, reduce_lr])
    return history


# Save the model
def save_load_model(model):
    '''
    save_load_model() saves the trained model in the .h5 format.
    '''
    # Save the trained model in the .h5 format
    model.save('genre_classification_model.h5')
    # Load the model for prediction
    model = load_model('genre_classification_model.h5')


# Function to predict the genre of an audio file
def predict_genre(audio_path, model, max_len=128):
    '''
    The predict_genre() function takes the path to an audio file as input and returns the predicted genre.
    1. Load the audio file using librosa.
    2. Extract the mel spectrogram features from the audio file.
    3. Convert the mel spectrogram to decibels.
    4. Pad the mel spectrogram with zeros if it is shorter than max_len.
    5. Reshape the mel spectrogram to match the input shape of the model.
    6. Normalize pixel values to [0, 1].
    7. Use the model to predict the genre.
    8. Return the predicted genre.
    '''
    try:
        y_, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
    # use librosa to extract mel spectrogram features from the audio file
    S = librosa.feature.melspectrogram(y=y_, sr=sr, n_mels=128, fmax=8000)
    # convert the mel spectrogram to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] < max_len:
        # pad the mel spectrogram with zeros if it is shorter than max_len
        pad_width = max_len - S_dB.shape[1]
        # use np.pad to pad the mel spectrogram with zeros on the right side
        S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :max_len]
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    feature = S_dB.reshape(1, 128, 128, 1)  # Reshape to match the input shape of the model
    feature = feature / 255.0  # Normalize pixel values to [0, 1]
    prediction = model.predict(feature)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre


def main():
    data_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/spectrograms'
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    features, labels = load_data(data_path, genres)

    features = features.reshape(features.shape[0], 128, 128, 1)  # Reshape to include channel dimension
    features = features.astype('float32') / 255.0  # Normalize pixel values to [0, 1] to improve model performance
    labels = LabelEncoder().fit_transform(labels)

    # Determine the input shape based on the flattened feature size
    input_shape = (128, 128, 1)
    num_genres = len(genres)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    # Create the model
    model = create_model(input_shape, num_genres)
    # Train the model
    model_train(model, X_train, y_train, X_val, y_val)
    # Save the model
    save_load_model(model)
    # Example usage
    audio_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/genres_original/jazz/jazz.00085.wav'
    predicted_genre = predict_genre(audio_path, model)
    if predicted_genre is not None:
        predicted_genre_label = genres[np.argmax(predicted_genre)]
        print(f'The predicted genre is: {predicted_genre_label}')
    else:
        print('Prediction failed.')


# Run the main function
if __name__ == "__main__":
    main()

