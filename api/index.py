from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your pre-trained Keras model
model = tf.keras.models.load_model('bestmodel.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the incoming JSON data
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    image_data = data['image']  # Extract the base64 image

    # Decode the base64 image
    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': 'Invalid base64 data'}), 400

    # Convert bytes to a PIL image
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image for your Keras model
    processed_image = preprocess_image(image)

    # Perform inference
    prediction = model.predict(processed_image)

    # Get the predicted class
    predicted_class = prediction.argmax(axis=1)[0]
    confidence = np.max(prediction)  # Get the confidence of the prediction

    # Mapping for tumor types
    if predicted_class == 0:
        result = 'Tumor Type: Glioma'
    elif predicted_class == 1:
        result = 'Tumor Type: Meningioma'
    elif predicted_class == 2:
        result = 'No Tumor Detected'
    else:
        result = 'Tumor Type: Pituitary'

    return jsonify({'result': result, 'confidence': float(confidence)})

def preprocess_image(image):
    image = image.resize((224, 224))  # Example size, modify as needed
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

if __name__ == '__main__':
    app.run(debug=True)
