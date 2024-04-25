# GoogleNet(Inception model)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
BATCH_SIZE = 32
EPOCHS = 20

# Define path to data directory
data_dir = 'drive/MyDrive/Colab Notebooks/newXray' # Dataset location

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

# Build the InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

# Add custom classification head
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Combine base model and custom head to create new model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // BATCH_SIZE)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator, verbose=1)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))
