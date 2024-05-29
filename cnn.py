import tkinter as tk
from tkinter import ttk
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Verify the shape of the data
print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')
print(f'Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}')

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(x_train)

# Build the improved CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Learning Rate Scheduler
lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                 patience=3, 
                                 verbose=1, 
                                 factor=0.5, 
                                 min_lr=0.00001)

class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        self.ax[0].set_xlabel('Epoch')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].set_title('Loss Plot')  # Add title to the loss plot
        self.ax[1].set_xlabel('Epoch')
        self.ax[1].set_ylabel('Accuracy')
        self.ax[1].set_title('Accuracy Plot')  # Add title to the accuracy plot
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epoch = 0
        self._model = None  # Note: use a private attribute

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        self.ax[0].cla()
        self.ax[0].plot(range(1, self.epoch + 1), self.train_loss, label='Train')
        self.ax[0].plot(range(1, self.epoch + 1), self.val_loss, label='Validation')
        self.ax[0].legend()
        self.ax[0].set_title('Loss Plot')  # Update title of the loss plot
        
        self.ax[1].cla()
        self.ax[1].plot(range(1, self.epoch + 1), self.train_acc, label='Train')
        self.ax[1].plot(range(1, self.epoch + 1), self.val_acc, label='Validation')
        self.ax[1].legend()
        self.ax[1].set_title('Accuracy Plot')  # Update title of the accuracy plot

        plt.tight_layout()
        plt.pause(0.1)  # Pause to allow GUI to update


# Create an instance of the PlotLosses class
plot_losses = PlotLosses()
plot_losses.model = model
# Create a simple Tkinter GUI
root = tk.Tk()
root.title('Training Progress')

def train_model():
    # Train the model with data augmentation
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=50, 
                        validation_data=(x_test, y_test),
                        callbacks=[lr_reduction, plot_losses])

train_button = ttk.Button(root, text='Train Model', command=train_model)
train_button.pack(pady=10)

root.mainloop()
