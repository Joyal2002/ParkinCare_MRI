from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
lesion_classifier = load_model("./parkinson first stage99.h5")
class_names = ['healthy', 'PD1']

def predict_image(image_path):
    img_array = cv2.imread(image_path)
    img_array = cv2.resize(img_array, (512, 512))

    img_yuv_1 = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_yuv = cv2.cvtColor(img_yuv_1, cv2.COLOR_RGB2YUV)

    y, u, v = cv2.split(img_yuv)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    y = clahe.apply(y)
    y = cv2.GaussianBlur(y, (3, 3), 1)

    img_array_1 = cv2.merge((y, u, v))
    img_array = cv2.cvtColor(img_array_1, cv2.COLOR_YUV2RGB)
    test_im = img_array.reshape(-1, 512, 512, 3)

    class_probabilities = lesion_classifier.predict(test_im)
    predicted_class_index = np.argmax(class_probabilities, axis=1)[0]

    return class_names[predicted_class_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
