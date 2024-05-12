from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('parkinson first stage99.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match model input shape
    image = image.resize((224, 224))
    # Convert image to numpy array
    image_array = np.asarray(image)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Expand dimensions to match model input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(file.read()))
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        result = prediction[0][0] * 100  # Assuming the model returns probability
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
