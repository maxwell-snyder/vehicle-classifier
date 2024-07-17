from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
from PIL import Image
import logging
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'my_flask_app/static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="vehicle_classifier_model_3.0_quantized.tflite")
interpreter.allocate_tensors()

# Assuming model expects input shape (1, 150, 150, 3)
def predict_image(img_path):
    img = Image.open(img_path)
    
    # Convert to RGB if image has alpha channel
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((150, 150))  # Resize image to target size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype(np.float32) / 255.0  # Rescale to [0, 1]

    # Perform inference with TensorFlow Lite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if the input shape matches what the model expects
    expected_shape = input_details[0]['shape']
    if not np.array_equal(img_array.shape, expected_shape):
        raise ValueError(f"Input shape mismatch: expected {expected_shape}, got {img_array.shape}")

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    class_indices = {'SUV': 0, 'Sedan': 1, 'Truck': 2}
    class_names = list(class_indices.keys())
    predicted_class = class_names[np.argmax(predictions)]
    rounded_predictions = np.round(predictions, decimals=5)  # Round to 5 decimal places
    return predicted_class, rounded_predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    app.logger.info('Received request with method: %s', request.method)
    if request.method == 'POST':
        if 'file' not in request.files:
            app.logger.warning('No file part in the request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            app.logger.warning('No selected file')
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)  # Save the uploaded file to UPLOAD_FOLDER
            predicted_class, predictions = predict_image(file_path)
            app.logger.info('Prediction: %s, Probabilities: %s', predicted_class, predictions)
            return render_template('index.html', prediction=predicted_class, probabilities=predictions.tolist(), image_path=file.filename)
    else:
        app.logger.info('Rendering the index page')
        
    return render_template('index.html', prediction=None, probabilities=None, image_path=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)