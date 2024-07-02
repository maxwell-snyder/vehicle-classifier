import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('vehicle_classifier_model.h5')
print("Model loaded from disk.")

# Make a prediction on a new image
img_path = r'C:\Users\max\OneDrive\My Coding Projects\VS code\projects\Vehicle detction\random testing images\david car.jpeg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Rescale to [0, 1]

predictions = model.predict(img_array)

# Round the prediction numbers to five decimal places
rounded_predictions = np.round(predictions, 5)

# Print the rounded predictions with five decimal places
print('Predictions (rounded):')
for i, pred in enumerate(rounded_predictions[0]):
    print(f'{pred:.5f}', end=' ')
print()

print('Sedan   SUV     Truck')

# Print the predictions in a more readable format
class_indices = {'sedan': 0, 'SUV': 1, 'truck': 2}  # Manually set class indices
class_names = list(class_indices.keys())
predicted_class = class_names[np.argmax(predictions)]
print(f'The model thinks this is a: {predicted_class}')




