import matplotlib.pyplot as plt

epochs_range = range(1, EPOCHS + 1)

plt.plot(epochs_range, history.history['accuracy'], '-o', label='Train')
plt.plot(epochs_range, history.history['val_accuracy'], '-x', label='Validation')

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()