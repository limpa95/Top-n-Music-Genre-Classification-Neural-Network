# Jun Seo

# Description: This script demonstrates how to train a neural network model to classify music genres using the GTZAN dataset.
# The model is trained on mel spectrogram features extracted from the audio files.
# The trained model is saved in the Keras format for later use.
# The script also demonstrates how to load the trained model and use it to predict the genre of a new audio file.
# The script uses the GTZAN dataset, which can be downloaded from http://marsyas.info/downloads/datasets.html.
# The dataset consists of 1000 audio files each of 30 seconds duration, belonging to 10 different genres.
# The genres are blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.
# The audio files are in WAV format and have a sampling rate of 22050 Hz and a bit depth of 16 bits.
# The mel spectrogram features are extracted using the librosa library.
# The model is a simple feedforward neural network with three dense layers.
# The input layer has 16 units, the hidden layer has 32 units, and the output layer has 10 units (one for each genre).
# The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
# The model is trained for 30 epochs with a batch size of 10.
# The trained model is saved in the Keras format using the model.save() method.
# The model is loaded using the load_model() function from the keras.models module.
# The predict_genre() function takes the path to an audio file as input and returns the predicted genre.
# The predict_tempo() function takes the path to an audio file as input and returns the predicted tempo in BPM.
# The tempo is estimated using the librosa library by calculating the onset strength and beat track.
# The tempo is then categorized into one of the following tempo markings: Grave, Largo, Adagio, Andante, Moderato, Allegro, Vivace, Presto.
# The tempo markings are based on the tempo ranges defined by the Italian musical terms.
# The predicted genre and tempo are printed to the console.
import numpy as np
import librosa
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#loading 5sec spectrogram data
def load_data(data_path, genres, img_size=(128, 128)):
    X = []
    y = []  # labels
    for genre in genres:
        genre_folder = os.path.join(data_path, genre)
        for img_file in os.listdir(genre_folder):
            if img_file.endswith('.png'):
                img_path = os.path.join(genre_folder, img_file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(img_size)
                img_array = np.array(img)
                X.append(img_array)
                y.append(genre)
    return np.array(X), np.array(y)

#data_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/genres_original'
data_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Top-n-Music-Genre-Classification-Neural-Network/5sec_spectrogram_2_4_2025'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
features, labels = load_data(data_path, genres)
# print(features)
#data_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/genres_original'
# Preprocess the features and labels
#features = np.array([f.flatten() for f in features])
features = features.reshape(features.shape[0], -1)  # Flatten the features
labels = LabelEncoder().fit_transform(labels)

# Determine the input shape based on the flattened feature size
input_shape = features.shape[1]
num_genres = len(genres)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
#create model
model = Sequential()
model.add(Dense(units=128, input_shape=(input_shape,), activation='relu',kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu',kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=num_genres, activation='softmax'))
# Compile the model
model.compile(optimizer=Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Reduce learning rate on plateau callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model with early stopping and learning rate reduction
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=100,
                    shuffle=True,
                    verbose=2,
                    callbacks=[early_stopping, reduce_lr])


# Save the trained model in the recommended Keras format
model.save('genre_classification_model.keras')

# Load the model for prediction
model = load_model('genre_classification_model.keras')


def predict_genre(audio_path, model, scaler, max_len=128):
    try:
        y_, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
    S = librosa.feature.melspectrogram(y=y_, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] < max_len:
        pad_width = max_len - S_dB.shape[1]
        S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        S_dB = S_dB[:, :max_len]
    feature = S_dB.flatten().reshape(1, -1)
    feature = scaler.transform(feature)
    prediction = model.predict(feature)
    predicted_genre = genres[np.argmax(prediction)]
    return predicted_genre


def predict_tempo(audio_path):
    try:
        y_, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
    onset_env = librosa.onset.onset_strength(y=y_, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return tempo


# Example usage
#audio_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/01 Twinkle, Twinkle, Little Star_ Var. A.wav'
audio_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/genres_original/jazz/jazz.00085.wav'
predicted_genre = predict_genre(audio_path, model, scaler)
if predicted_genre:
    print(f'The predicted genre is: {predicted_genre}')
else:
    print('Prediction failed.')

#Grave = 20-40 BPM, Largo = 40-60 BPM, Adagio = 60-76 BPM, Andante = 76-108 BPM, Moderato = 108-120 BPM, Allegro = 120-156 BPM, Vivace 156-176, Presto = 176-200 BPM    
tempo_category = ['Grave', 'Largo', 'Adagio', 'Andante', 'Moderato', 'Allegro', 'Vivace', 'Presto']
predicted_tempo = predict_tempo(audio_path)
predicted_tempo_marking = None
if predicted_tempo is not None:
    predicted_tempo = predicted_tempo.item()  # Extract the scalar value from the numpy array
    if predicted_tempo < 40:
        predicted_tempo_marking = 'Grave'
    elif predicted_tempo < 60:
        predicted_tempo_marking = 'Largo'
    elif predicted_tempo < 76:
        predicted_tempo_marking = 'Adagio'
    elif predicted_tempo < 108:
        predicted_tempo_marking = 'Andante'
    elif predicted_tempo < 120:
        predicted_tempo_marking = 'Moderato'
    elif predicted_tempo < 156:
        predicted_tempo_marking = 'Allegro'
    elif predicted_tempo < 176:
        predicted_tempo_marking = 'Vivace'
    else:
        predicted_tempo_marking = 'Presto'
    predicted_tempo = round(predicted_tempo)
    print(f'The predicted tempo is: {predicted_tempo} BPM, ({predicted_tempo_marking})')
else:
    print('Tempo prediction failed.')

    
