# This is an example of a Flask API that will be deployed to
# an AWS backend server. This API loads the trained model
# (the .h5 file) and creates a /predict endpoint that accepts
#  a JPEG file via a POST request to the AWS server. In the
# future, a zip file may be used with multiple jpegs instead
# of one at a time. The API processes the sent image using the
# same preprocessing steps as during training (resizing to
# 128×128 and normalizing it to the right specifications) and
# returns the prediction results.

# Note: This will only work if the model file (.h5) correctly predicts and
# is available on the AWS server in the correct location (or update the file
# path accordingly under MODEL_SAVE_FILE). Since the frontend and backend
# may be on different domains, the API uses CORS to allow cross-origin
# requests.


from flask import Flask, request, jsonify
from flask_cors import CORS  # type: ignore
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
HOST = "0.0.0.0"  # change IP address and PORT based on server
PORT = 5000
IMG_SIZE = (128, 128)
MODEL_SAVE_FILE = "model/genre_classification_cnn.h5"
genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

app = Flask(__name__)
CORS(app)

# Loads the trained model once during startup
model = tf.keras.models.load_model(MODEL_SAVE_FILE)


def preprocess_image(image):
    """
    Preprocess the input jpeg image so that it matches the format
    used during training:
      - Convert to RGB
      - Resize to IMG_SIZE (128, 128)
      - Normalize pixel values to [0, 1]
    """
    image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """
    This endpoint accepts a JPEG image file via a POST request.
    The file should be sent with the form-data key 'file'.
    The image is preprocessed and passed to the model for prediction.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file found in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file.stream)
        # Preprocess
        img_array = preprocess_image(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        prediction_val = predictions[0]

        # Convert to JSON
        prediction_map = {
            genres[i]: float(prediction_val[i]) for i in range(len(genres))
        }

        return jsonify(prediction_map)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':

    app.run(host=HOST, port=PORT)
