import seaborn as sns
import matplotlib as plt
import tensorflow as tf
import numpy as np

# Make predictions on validation data
predictions = np.argmax(model.predict(validation_data), axis=-1)
labels = validation_data.classes

# Compute confusion matrix
conf = tf.math.confusion_matrix(labels=labels, predictions=predictions)

# Plot confusion matrix as a heatmap
sns.heatmap(conf, annot=True, cmap='Blues', yticklabels=validation_data.class_indices.keys(), xticklabels=validation_data.class_indices.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
