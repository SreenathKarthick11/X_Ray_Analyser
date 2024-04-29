# VGG16 MODEL

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16

# Define constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
BATCH_SIZE = 32
EPOCHS = 20

# Define path to data directory
data_dir = 'drive/MyDrive/Colab Notebooks/newXray' # dataset location

# Define data generator with validation split
gendata = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             validation_split=0.20)

train_data = gendata.flow_from_directory(data_dir,
                                              target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                              batch_size=BATCH_SIZE,
                                              class_mode='categorical',
                                              subset='training')  # Training data

# Generate validation dataset
validation_data = gendata.flow_from_directory(data_dir,
                                                   target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical',
                                                   subset='validation')  # Validation data

# Build the VGG model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Freeze the base_model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_data.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=validation_data,
                    validation_steps=validation_data.samples // BATCH_SIZE)

# Evaluate the model
loss, accuracy = model.evaluate(validation_data, verbose=1)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Load and preprocess the image
img_path = 'drive/MyDrive/Colab Notebooks/newXray/Fractured Arm/IMG0000057.jpg' # image location
img = image.load_img(img_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to match model input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
predictions = model.predict(img_array)

# Interpret the predictions
class_indices = train_data.class_indices
predicted_class = list(class_indices.keys())[np.argmax(predictions)]

print("Predicted class:", predicted_class)
