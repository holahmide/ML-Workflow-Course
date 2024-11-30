from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from werkzeug.utils import secure_filename

# Load the model
# model = joblib.load('/Users/olamideadeniyi/Documents/Masters/TME6015/Project/Code/backend/model.pkl')

# Initialize Flask app
app = Flask(__name__)

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
                # Preprocess the image
                img = img.resize((224, 224))  # Resize to the input size expected by your model
                img_array = np.array(img)    # Convert to NumPy array

                # Normalize pixel values (example: scale to 0-1 if needed by the model)
                img_array = img_array / 255.0

                # Add batch dimension (1, height, width, channels)
                img_array = np.expand_dims(img_array, axis=0)

                # Perform prediction
                # predictions = model.predict(img_array).tolist()
                predictions = img_array.tolist()

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
