from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model
model = load_model('vehicle_classifier_model.h5')

# Function to predict the class of the uploaded image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale to [0, 1]

    predictions = model.predict(img_array)
    class_indices = {'sedan': 0, 'SUV': 1, 'truck': 2}
    class_names = list(class_indices.keys())
    predicted_class = class_names[np.argmax(predictions)]
    rounded_predictions = np.round(predictions, decimals=5)  # Round to 5 decimal places
    return predicted_class, rounded_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)  # Save the uploaded file to UPLOAD_FOLDER
            predicted_class, predictions = predict_image(file_path)
            return render_template('index.html', prediction=predicted_class, probabilities=predictions.tolist(), image_path=file.filename)

    return render_template('index.html', prediction=None, probabilities=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
   
