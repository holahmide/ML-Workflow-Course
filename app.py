from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


# Loading the model
MODEL_PATH = os.path.join(os.getcwd(), 'backend/traffic_light_model.h5')
model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)



def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Traffic Light" if prediction[0][0] < 0.5 else "No Traffic Light"

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
            # Open the image
            with Image.open(file_path) as img:
                predictions = predict_image(file_path)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload a PNG, JPG, or JPEG image.'}), 400

        # Clean up the temporary file
        os.remove(file_path)

        # Return the predictions
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
