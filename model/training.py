# Responsible for training the model to use the spectrogram dataset to classify the audio genres.  The 
# data is located within the data/images_original directory.  The model is saved to the model directory.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# define the genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
data_dir = 'data/images_original'

# split the data into training and validation sets
img_height = 180
img_width = 180
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    label_mode='categorical', 
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create the model
input_shape = (img_height, img_width, 3)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Helps reduce overfitting
    tf.keras.layers.Dense(len(genres), activation='softmax')
])


# Compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20  
)

# Evaluation
test_loss, test_accuracy = model.evaluate(val_ds, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

model.save('model/audio_NN_model.h5')