# 🦁 **Multi-Class Animal Recognition for Wildlife Conservation** 🌿

## Overview 🌍

This project focuses on using AI and deep learning techniques for wildlife conservation, specifically through multi-class animal recognition. The goal is to automate the identification of animal species from images, enabling more efficient monitoring and protection of wildlife. By leveraging **Convolutional Neural Networks (CNNs)**, we can identify species in real-time with high accuracy, significantly reducing manual labor and human error in wildlife tracking.

## Tools & Technologies 🛠️

- **Programming Language**: Python 🐍
- **Deep Learning Libraries**: TensorFlow, Keras 🤖
- **Data Handling**: NumPy, Pandas 📊
- **Image Processing**: OpenCV, Matplotlib 📸
- **Cloud Environment**: Google Colab (GPU enabled for faster training) ☁️
- **Dataset**: The Wildfire Dataset 🐾 (Contains images of animals categorized into multiple classes)

## Dataset 📂

The dataset used for this project is **The Wildfire Dataset**, which contains labeled images of animals in various categories. The images are organized into three directories:

- **Train Directory**: Contains images used for training the model.
- **Validation Directory**: Used for validating the model's performance during training.
- **Test Directory**: Used to evaluate the model after training.

Classes in the dataset:
- `nofire`
- `fire`

### Path to Dataset:
```
/kaggle/input/the-wildfire-dataset
```

## Workflow ⚙️

### 1. **Data Preparation** 🗂️
The data is first downloaded and organized into directories for training, validation, and testing. Each class is stored in a separate folder within the directory.

### 2. **Preprocessing** 🔄
- Image resizing and normalization are applied to the images to make them suitable for feeding into the deep learning model.
- Data augmentation is employed to create variations of images, increasing the model's ability to generalize.

### 3. **Model Architecture** 🏗️
A **Convolutional Neural Network (CNN)** is built for image classification:
- **Convolution Layers** for feature extraction.
- **Max-Pooling** to reduce dimensionality.
- **Fully Connected Layers** for classification.

### 4. **Training** 🏋️‍♂️
The model is trained using the preprocessed data, with the following parameters:
- **Batch Size**: 32
- **Image Dimensions**: 150x150 pixels
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy (for binary classification)

### 5. **Evaluation & Testing** 🧑‍🔬
The trained model is evaluated on unseen test data. The model's performance is measured using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

## Key Features ✨

- **GPU Acceleration**: The model is trained using GPU for faster processing, ensuring quicker results. ⚡
- **Real-time Prediction**: The trained model can be used for real-time predictions on new animal images 🖼️.
- **Scalable**: The model can be extended to include additional species as new data becomes available 🌱.

## Installation & Setup ⚙️

### Prerequisites 📦

- Python 3.x 🐍
- TensorFlow (for model training) 🤖
- Keras (for building and training the model) ⚙️
- Matplotlib (for visualization) 📈
- Kaggle API (for downloading datasets) 📥

### Installation

```bash
pip install tensorflow keras matplotlib kaggle
```

### Running the Code ▶️

1. Download the dataset using Kaggle API:
```python
import kagglehub
path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")
print(f"Path to dataset files: {path}")
```

2. Import necessary libraries:
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
```

3. Prepare and preprocess the dataset:
```python
train_dir = '/path/to/train'
val_dir = '/path/to/val'
test_dir = '/path/to/test'
```

4. Visualize sample images from the dataset:
```python
plt.figure(figsize=(12, 10))
for i in range(5):
  img_path = os.path.join(train_dir, 'fire', os.listdir(os.path.join(train_dir, 'fire'))[i])
  img = plt.imread(img_path)
  plt.subplot(1, 5, i+1)
  plt.imshow(img)
  plt.title(f'fire \\n shape: {img.shape}')
  plt.axis('off')
plt.show()
```

5. Define the CNN model architecture and train:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

6. Evaluate and test the model:
```python
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=32, class_mode='binary', target_size=(150, 150))
val_generator = val_datagen.flow_from_directory(val_dir, batch_size=32, class_mode='binary', target_size=(150, 150))
```

## Conclusion 🌟

The model built in this project successfully identifies animal species based on images, which supports wildlife conservation efforts by automating the process of monitoring animal populations. This reduces human error and manual effort while enabling faster and more accurate decision-making. The model can be further enhanced with additional species and used for real-time conservation monitoring 🌿.

---
