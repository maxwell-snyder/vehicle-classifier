import tensorflow as tf

# Path to the original .h5 model
h5_model_path = 'vehicle_classifier_model_3.0.h5'

# Path to save the quantized TFLite model
tflite_model_path = 'vehicle_classifier_model_3.0_quantized.tflite'

# Load the .h5 model
model = tf.keras.models.load_model(h5_model_path)

# Ensure the model is built before converting
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create a converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to TFLite format with quantization
tflite_model = converter.convert()

# Save the quantized model to disk
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Quantized model saved to disk.")
