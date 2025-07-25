# 🧠 Deep Learning for Cancer Cell Classification

A Convolutional Neural Network (CNN)-based system designed to classify cancer cell images into three distinct categories. The project focuses on preprocessing, model training, and evaluation using real-world histopathological image data.

---

## 🧪 Project Overview

- Implemented using TensorFlow and Keras
- Three-class image classification task
- Applied standard preprocessing and model evaluation techniques
- Designed for medical image classification and analysis use case

---

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Matplotlib / Seaborn
- scikit-learn
- joblib

---

## 🧠 Model Architecture

CNN-based Sequential Model:
```python
input_shape = (244, 244, 3)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
