from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all origins
CORS(app)

# Path to the model
MODEL_PATH = os.path.join(os.getcwd(), 'traffic_light_model.h5')

# Load the model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Helper function to predict an image
def predict_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))  # Adjust target size if needed
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return "Traffic Light" if prediction[0][0] < 0.5 else "No Traffic Light"
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Ensure it's a valid file
    if file.filename == '':
        return jsonify({'error': 'Empty file name provided'}), 400

    try:
        # Secure and save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file.save(file_path)

        # Process the file as an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            predictions = predict_image(file_path)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload a PNG, JPG, or JPEG image.'}), 400

        # Clean up the temporary file
        os.remove(file_path)

        # Return the predictions
        return jsonify({'predictions': predictions})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Use a specific host and port for easier deployment
    app.run(host='0.0.0.0', port=5000, debug=True)
