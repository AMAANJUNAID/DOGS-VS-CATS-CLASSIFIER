import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
# Step 1: Extract Main Zip File
main_zip_path = 'C:/Users/USER/Downloads/ML tasks/3rd  task/dogs-vs-cats.zip'  # Replace with the actual path of your dataset.zip file
main_extract_path = 'C:/Users/USER/Downloads/ML tasks/3rd  task/train'  # Folder where data will be extracted

if not os.path.exists(main_extract_path):
    with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
        zip_ref.extractall(main_extract_path)
    print(f"Main dataset extracted to: {main_extract_path}")
else:
    print(f"Dataset already extracted at: {main_extract_path}")

# Step 2: Define Paths for Train and Test Data
train_dir = os.path.join(main_extract_path, 'C:/Users/USER/Downloads/ML tasks/3rd  task/train')  # Adjust folder name as per your dataset structure
test_dir = os.path.join(main_extract_path, 'C:/Users/USER/Downloads/ML tasks/3rd  task/test1')    # Adjust folder name as per your dataset structure
# Check if paths exist and display contents
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Train or Test directory not found! Check dataset structure.")

print(f"Train Directory Content: {os.listdir(train_dir)[:5]}")
print(f"Test Directory Content: {os.listdir(test_dir)[:5]}")


# Step 4: Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split training data into training and validation sets
)


# Training Data Generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation Data Generator
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


# Test Data Generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # Testing data typically does not have labels
    shuffle=False
)


# Step 5: Build the Model
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])


# Step 6: Compile the Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Step 8: Save the Model
model.save('dogs_vs_cats_classifier.keras')
print("Model saved as 'dogs_vs_cats_classifier.h5'")


# Step 9: Plot Training and Validation Results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# Ensure test_generator is initialized correctly
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),  # Adjust to match your model's input size
    batch_size=32,
    class_mode='binary'
)

# Debugging: Verify the generator has loaded data
print("Number of samples in test_generator:", test_generator.samples)
if test_generator.samples == 0:
    raise ValueError("No samples found in the test directory. Please check test_dir and its structure.")

# Reset generator (optional)
test_generator.reset()

# Make predictions
predictions = model.predict(test_generator)

# Convert predictions to classes
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]


# Step 11: Save Predictions to CSV
filenames = test_generator.filenames
results = pd.DataFrame({
    'Filename': filenames,
    'Prediction': predicted_classes
})
results.to_csv('test_predictions.csv', index=False)
print("Predictions saved to 'test_predictions.csv'")