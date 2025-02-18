from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
import ssl
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
   
# Disable SSL verification for downloading pre-trained models, if needed
ssl._create_default_https_context = ssl._create_unverified_context
app = Flask(__name__)
  
  
# Load the pre-trained model.
model = tf.keras.models.load_model('./model/model.h5', compile=False)

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    """
    Preprocesses the image for model prediction.

    Args:
        file_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    # Load the image
    img = load_img(file_path, target_size=(64, 64))  # Resize to match the model's input size
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to predict the blood group from fingerprint image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img = preprocess_image(file_path)

        # Perform prediction
        predictions = model.predict(img)

        predicted_class = int(np.argmax(predictions[0]))
        print('predicted_class is:', predicted_class)

        # Optional: Define class names (if not in the model)
        class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']  # Example classes

        predicted_label = class_names[predicted_class]

        # Return the prediction as json

        return jsonify({'predicted_class': predicted_class,
                        'predicted_label':predicted_label,
                        'confidence':float(np.max(predictions[0]))
                        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up: remove the saved file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '_main_':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
