import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: sedan, SUV, truck
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Paths to the training and validation directories
train_dir = r"C:\Users\max\OneDrive\My Coding Projects\VS code\projects\Vehicle detction\dataset\train"
val_dir = r"C:\Users\max\OneDrive\My Coding Projects\VS code\projects\Vehicle detction\dataset\validation"

# Create an instance of ImageDataGenerator for training data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values to [0, 1]
    shear_range=0.2,      # Apply shear transformations
    zoom_range=0.2,       # Apply zoom transformations
    horizontal_flip=True  # Randomly flip images horizontally
)

# Create an instance of ImageDataGenerator for validation data (without data augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale pixel values

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,            # Path to the training data directory
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,        # Number of images to return in each batch
    class_mode='categorical'   # Use 'categorical' for multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,              # Path to the validation data directory
    target_size=(150, 150),  # Resize images to 150x150 pixels
    batch_size=32,        # Number of images to return in each batch
    class_mode='categorical'   # Use 'categorical' for multi-class classification
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Save the model
model.save('vehicle_classifier_model.h5')
print("Model saved to disk.")

