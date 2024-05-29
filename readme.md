# Convolutional Neural Network (CNN) for Image Classification

## Introduction

This Python script `cnn.py` implements a Convolutional Neural Network (CNN) using TensorFlow and Keras for image classification. The script utilizes the CIFAR-10 dataset, which is a benchmark dataset for image classification tasks. 

## Features

- Loads the CIFAR-10 dataset and preprocesses the data.
- Implements data augmentation using Keras's `ImageDataGenerator`.
- Constructs a CNN model using convolutional layers, batch normalization, max-pooling, and fully connected layers with dropout regularization.
- Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.
- Implements a learning rate scheduler to dynamically adjust the learning rate during training.
- Trains the model using the augmented data generator and evaluates its performance on the test set.
- Visualizes the training and validation accuracy over epochs.

## Usage

1. **Clone the Repository:**

    ```
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2. **Install Dependencies:**

    ```
    pip install tensorflow matplotlib numpy
    ```

3. **Run the Script:**

    ```
    python cnn.py
    ```

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib
- NumPy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The CIFAR-10 dataset is obtained from the Keras datasets module.
- TensorFlow and Keras provide the deep learning framework for building and training the CNN model.
