from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(data_path, genres, max_len=128):
    X = []
    y = []  # labels
    # Iterate through the dataset and print the audio and label of each batch
    for genre in genres:
        genre_folder = os.path.join(data_path, genre)
        for audio_file in os.listdir(genre_folder):
            audio_path = os.path.join(genre_folder, audio_file)
            y_, sr = librosa.load(audio_path)
            S = librosa.feature.melspectrogram(y=y_, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            if S_dB.shape[1] < max_len:
                pad_width = max_len - S_dB.shape[1]
                S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                S_dB = S_dB[:, :max_len]
            X.append(S_dB)
            y.append(genre)
    return X, y


data_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/genres_original'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
features, labels = load_data(data_path, genres)
# print(features)
data_path = '/Users/junseo/Desktop/OSU/9th term/CS467 Capstone Project/project/Data/genres_original'
# Preprocess the features and labels
features = np.array([f.flatten() for f in features])
labels = LabelEncoder().fit_transform(labels)

# Determine the input shape based on the flattened feature size
input_shape = features.shape[1]

model = Sequential([
    Dense(units=16, input_shape=(input_shape,), activation='relu'),  # Adjust input shape
    Dense(units=32, activation='relu'),
    Dense(units=10, activation='softmax')  # Adjust output units to match the number of genres
])

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train the model
model.fit(x=X_train,
          y=y_train,
          validation_data=(X_val, y_val),
          batch_size=10,
          epochs=30,
          shuffle=True,
          verbose=2)

