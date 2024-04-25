# Final CNN MODEL

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation , Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Define constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
BATCH_SIZE = 32
EPOCHS = 20


# Define path to data directory
data_dir= 'drive/MyDrive/Colab Notebooks/newXray' # file location

# Define data generator with validation split
datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             validation_split=0.20)

train_generator = datagen.flow_from_directory(data_dir,
                                              target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                              batch_size=BATCH_SIZE,
                                              class_mode='categorical',
                                              subset='training')  # Training data

# Generate validation dataset
validation_generator = datagen.flow_from_directory(data_dir,
                                                   target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode='categorical',
                                                   subset='validation')  # Validation data

# early_stopping_callback = EarlyStopping(monitor="val_loss",patience = 2)
# callbacks = early_stopping_callback


# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // BATCH_SIZE
                    )

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator, verbose=1)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Load and preprocess the image
img_path = 'drive/MyDrive/Colab Notebooks/newXray/Fractured Arm/IMG0000057.jpg' # Image location
img = image.load_img(img_path, target_size=(150, 150))  # Resize to match model input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
predictions = model.predict(img_array)

# Interpret the predictions
class_indices = train_generator.class_indices
predicted_class = list(class_indices.keys())[np.argmax(predictions)]

print("Predicted class:", predicted_class)