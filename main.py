# prompt: #import drive

from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.tree import DecisionTreeClassifier

# Step 1: Extract the dataset
zip_file_path = '/content/drive/MyDrive/BDPapayaLeaf.zip'  # Update with your zip file path
extracted_folder = '/content/file'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Step 2: Define paths and labels
image_size = (128, 128)
folder_path = os.path.join(extracted_folder, 'BDPapayaLeaf', 'Original Images')
label_map = {'Anthracnose': 0, 'BacterialSpot': 1, 'Curl': 2, 'Healthy': 3, 'RingSpot': 4}

# Step 3: Load and preprocess images
images, labels = [], []

for label_name, label in label_map.items():
    folder = os.path.join(folder_path, label_name)
    for img_name in tqdm(os.listdir(folder), desc=f"Processing {label_name}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)

# Step 4: Convert to numpy arrays and normalize
X = np.array(images) / 255.0  # Normalize
y = np.array(labels)

# Step 5: Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Data augmentation for CNN
train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
train_datagen.fit(X_train)

# Step 7: CNN Model Definition
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 output classes
])

# Step 8: Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Step 9: Train the CNN model
cnn_history = cnn_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Step 10: Evaluate the CNN model
cnn_train_acc = cnn_model.evaluate(X_train, y_train)[1]
cnn_val_acc = cnn_model.evaluate(X_val, y_val)[1]

print(f'CNN Training Accuracy: {cnn_train_acc * 100:.2f}%')
print(f'CNN Validation Accuracy: {cnn_val_acc * 100:.2f}%')

# Step 11: Plot CNN accuracy and loss
plt.plot(cnn_history.history['accuracy'], label="Training Accuracy")
plt.plot(cnn_history.history['val_accuracy'], label="Validation Accuracy")
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(cnn_history.history['loss'], label="Training Loss")
plt.plot(cnn_history.history['val_loss'], label="Validation Loss")
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 12: Confusion Matrix and Classification Report for CNN
cnn_preds = np.argmax(cnn_model.predict(X_val), axis=1)
cnn_cm = confusion_matrix(y_val, cnn_preds)

plt.figure(figsize=(10, 7))
sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("CNN Classification Report:")
print(classification_report(y_val, cnn_preds, target_names=label_map.keys()))

# ---------------------------
# Decision Tree Model (For Comparison)
# ---------------------------

# Step 13: Flatten images for Decision Tree
X_flattened = X.reshape(X.shape[0], -1)

# Step 14: Split data for Decision Tree
X_train_flat, X_val_flat, y_train_flat, y_val_flat = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# Step 15: Train Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_flat, y_train_flat)

# Step 16: Evaluate Decision Tree
tree_train_acc = tree_model.score(X_train_flat, y_train_flat)
tree_val_acc = tree_model.score(X_val_flat, y_val_flat)

print(f'Decision Tree Training Accuracy: {tree_train_acc * 100:.2f}%')
print(f'Decision Tree Validation Accuracy: {tree_val_acc * 100:.2f}%')

import zipfile
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# -----------------------------
# Step 1: Extract the dataset
zip_file_path = '/content/drive/MyDrive/BDPapayaLeaf.zip'  # Update with your zip file path
extracted_folder = '/content/file'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Step 2: Define paths and labels
image_size = (128, 128)
folder_path = os.path.join(extracted_folder, 'BDPapayaLeaf', 'Original Images')
label_map = {'Anthracnose': 0, 'BacterialSpot': 1, 'Curl': 2, 'Healthy': 3, 'RingSpot': 4}

# Step 3: Load and preprocess images
images, labels = [], []

for label_name, label in label_map.items():
    folder = os.path.join(folder_path, label_name)
    for img_name in tqdm(os.listdir(folder), desc=f"Processing {label_name}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)

# Step 4: Convert to numpy arrays and normalize
X = np.array(images) / 255.0  # Normalize
y = np.array(labels)

# Step 5: Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Data augmentation for CNN
train_datagen = ImageDataGenerator(
    rotation_range=30,  # Increased rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # Fill mode for missing pixels
)
train_datagen.fit(X_train)

# Step 7: CNN Model Definition
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 output classes
])

# Step 8: Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Step 9: Train the CNN model
cnn_history = cnn_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,  # Increased number of epochs
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Step 10: Evaluate the CNN model
cnn_train_acc = cnn_model.evaluate(X_train, y_train)[1]
cnn_val_acc = cnn_model.evaluate(X_val, y_val)[1]

print(f'CNN Training Accuracy: {cnn_train_acc * 100:.2f}%')
print(f'CNN Validation Accuracy: {cnn_val_acc * 100:.2f}%')

# Step 11: Plot CNN accuracy and loss
plt.plot(cnn_history.history['accuracy'], label="Training Accuracy")
plt.plot(cnn_history.history['val_accuracy'], label="Validation Accuracy")
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(cnn_history.history['loss'], label="Training Loss")
plt.plot(cnn_history.history['val_loss'], label="Validation Loss")
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 12: Confusion Matrix and Classification Report for CNN
cnn_preds = np.argmax(cnn_model.predict(X_val), axis=1)
cnn_cm = confusion_matrix(y_val, cnn_preds)

plt.figure(figsize=(10, 7))
sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("CNN Classification Report:")
print(classification_report(y_val, cnn_preds, target_names=label_map.keys()))

# ----------------------------------------
# Part 2: Predict and Classify Images
# ----------------------------------------

# Dictionary to store disease details and prevention
disease_info = {
    'Anthracnose': {
        'details': 'Anthracnose is caused by various fungal pathogens affecting many plant species. Symptoms include dark, sunken lesions on leaves and stems.',
        'prevention': 'Remove infected plant debris, avoid overhead watering, and apply fungicides as needed.'
    },
    'Bacterial Spot': {
        'details': 'Bacterial Spot is characterized by water-soaked spots on leaves, which may lead to leaf drop.',
        'prevention': 'Rotate crops, use disease-resistant varieties, and practice good sanitation.'
    },
    'Curl': {
        'details': 'Curl is often a viral infection causing leaf distortion and curling.',
        'prevention': 'Control aphids and other pests that spread viruses, and remove infected plants.'
    },
    'Healthy': {
        'details': 'The plant is healthy and free from diseases.',
        'prevention': 'Maintain good cultural practices, including proper watering, fertilization, and pruning.'
    },
    'Ring Spot': {
        'details': 'Ring Spot is caused by viral infections characterized by ring-like patterns on leaves.',
        'prevention': 'Control aphids and remove infected plants to prevent the spread of the virus.'
    }
}

import numpy as np
import cv2
import tensorflow as tf

# Define the disease information
disease_info = {
    0: {
        "name": "Anthracnose",
        "details": "Anthracnose causes dark, sunken lesions on leaves and stems, which may cause premature leaf drop.",
        "prevention": "Prune infected areas, avoid overhead irrigation, and apply fungicides if necessary."
    },
    1: {
        "name": "Bacterial Spot",
        "details": "Bacterial Spot is characterized by water-soaked spots on leaves, which may lead to leaf drop.",
        "prevention": "Rotate crops, use disease-resistant varieties, and practice good sanitation."
    },
    2: {
        "name": "Curl",
        "details": "Curl diseases cause distorted, puckered leaves that are curled or cupped upwards.",
        "prevention": "Apply copper-based fungicides during dormant seasons and remove infected plant material."
    },
    3: {
        "name": "Healthy",
        "details": "The plant is healthy with no visible signs of disease.",
        "prevention": "Continue good gardening practices, including proper watering and pest control."
    },
    4: {
        "name": "Ring Spot",
        "details": "Ring Spot shows up as circular yellow or brown lesions on leaves, which can merge and cause defoliation.",
        "prevention": "Remove infected plants, use virus-free seeds, and control aphid populations."
    }
}

# Assuming you have already trained the model and loaded it
# Example of loading a pre-trained model (modify the path as needed)
model = tf.keras.models.load_model('/content/drive/MyDrive/plant_disease_model.h5')

# Example of loading an image for prediction (modify the image path)
image_path = '/content/Screenshot 2024-10-13 004401.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))  # Resize to match the input size of the model
image = image / 255.0  # Normalize the image

# Predict the disease from the image
predictions = np.argmax(model.predict(np.expand_dims(image, axis=0)), axis=1)
predicted_class = predictions[0]  # Get the predicted class (0 to 4)

# Fetch the corresponding disease information
disease_name = disease_info[predicted_class]['name']
disease_details = disease_info[predicted_class]['details']
disease_prevention = disease_info[predicted_class]['prevention']

# Display the output
print(f"Disease: {disease_name}")
print(f"Details: {disease_details}")
print(f"Prevention: {disease_prevention}")
# Import libraries
from google.colab import drive
import zipfile
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.tree import DecisionTreeClassifier

# Step 1: Mount Google Drive and extract the dataset
drive.mount('/content/drive')
zip_file_path = '/content/drive/MyDrive/BDPapayaLeaf.zip'  # Path to the dataset
extracted_folder = '/content/file'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# Step 2: Define paths and labels
image_size = (128, 128)
folder_path = os.path.join(extracted_folder, 'BDPapayaLeaf', 'Original Images')
label_map = {'Anthracnose': 0, 'BacterialSpot': 1, 'Curl': 2, 'Healthy': 3, 'RingSpot': 4}

# Step 3: Load and preprocess images
images, labels = [], []
for label_name, label in label_map.items():
    folder = os.path.join(folder_path, label_name)
    for img_name in tqdm(os.listdir(folder), desc=f"Processing {label_name}"):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(label)

# Step 4: Convert to numpy arrays and normalize
X = np.array(images) / 255.0  # Normalize
y = np.array(labels)

# Step 5: Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Data augmentation for CNN
train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
train_datagen.fit(X_train)

# Step 7: CNN Model Definition
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 output classes
])

# Step 8: Compile the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Step 9: Train the CNN model
cnn_history = cnn_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Step 10: Evaluate the CNN model
cnn_train_acc = cnn_model.evaluate(X_train, y_train)[1]
cnn_val_acc = cnn_model.evaluate(X_val, y_val)[1]

print(f'CNN Training Accuracy: {cnn_train_acc * 100:.2f}%')
print(f'CNN Validation Accuracy: {cnn_val_acc * 100:.2f}%')

# Step 11: Plot CNN accuracy and loss
plt.plot(cnn_history.history['accuracy'], label="Training Accuracy")
plt.plot(cnn_history.history['val_accuracy'], label="Validation Accuracy")
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(cnn_history.history['loss'], label="Training Loss")
plt.plot(cnn_history.history['val_loss'], label="Validation Loss")
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 12: Confusion Matrix and Classification Report for CNN
cnn_preds = np.argmax(cnn_model.predict(X_val), axis=1)
cnn_cm = confusion_matrix(y_val, cnn_preds)

plt.figure(figsize=(10, 7))
sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("CNN Classification Report:")
print(classification_report(y_val, cnn_preds, target_names=label_map.keys()))

# ---------------------------
# Decision Tree Model (For Comparison)
# ---------------------------

# Step 13: Flatten images for Decision Tree
X_flattened = X.reshape(X.shape[0], -1)

# Step 14: Split data for Decision Tree
X_train_flat, X_val_flat, y_train_flat, y_val_flat = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# Step 15: Train Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_flat, y_train_flat)

# Step 16: Evaluate Decision Tree
tree_train_acc = tree_model.score(X_train_flat, y_train_flat)
tree_val_acc = tree_model.score(X_val_flat, y_val_flat)

print(f'Decision Tree Training Accuracy: {tree_train_acc * 100:.2f}%')
print(f'Decision Tree Validation Accuracy: {tree_val_acc * 100:.2f}%')

# Dictionary to store disease details and prevention
disease_info = {
    'Anthracnose': {
        'details': 'Anthracnose is caused by fungal pathogens. Symptoms include dark lesions on leaves.',
        'prevention': 'Remove infected plant debris and apply fungicides.'
    },
    'Bacterial Spot': {
        'details': 'Bacterial Spot causes water-soaked spots on leaves.',
        'prevention': 'Rotate crops and use disease-resistant varieties.'
    },
    'Curl': {
        'details': 'Curl causes leaf distortion, often viral in nature.',
        'prevention': 'Control pests and remove infected plants.'
    },
    'Healthy': {
        'details': 'The plant is healthy and free from diseases.',
        'prevention': 'Maintain proper watering and fertilization.'
    },
    'Ring Spot': {
        'details': 'Ring Spot causes viral infections characterized by ring-like patterns.',
        'prevention': 'Control aphids and remove infected plants.'
    }
}

# Function to classify a new image
def classify_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    predictions = cnn_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Get disease name
    disease_name = list(disease_info.keys())[predicted_class]
    return disease_name, disease_info[disease_name]

# Example usage
image_path = '/content/Screenshot 2024-10-13 004401.png'  # Replace with the path to your test image
disease_name, info = classify_image(image_path)
print(f"Disease: {disease_name}")
print(f"Details: {info['details']}")
print(f"Prevention: {info['prevention']}")



image_path = '/content/Screenshot 2024-10-13 005411.png'  # Replace with the path to your test image
disease_name, info = classify_image(image_path)
print(f"Disease: {disease_name}")
print(f"Details: {info['details']}")
print(f"Prevention: {info['prevention']}")
import numpy as np
import cv2
import tensorflow as tf

# Define the disease information
disease_info = {
    0: {
        "name": "Anthracnose",
        "details": "Anthracnose causes dark, sunken lesions on leaves and stems, which may cause premature leaf drop.",
        "prevention": "Prune infected areas, avoid overhead irrigation, and apply fungicides if necessary."
    },
    1: {
        "name": "Bacterial Spot",
        "details": "Bacterial Spot is characterized by water-soaked spots on leaves, which may lead to leaf drop.",
        "prevention": "Rotate crops, use disease-resistant varieties, and practice good sanitation."
    },
    2: {
        "name": "Curl",
        "details": "Curl diseases cause distorted, puckered leaves that are curled or cupped upwards.",
        "prevention": "Apply copper-based fungicides during dormant seasons and remove infected plant material."
    },
    3: {
        "name": "Healthy",
        "details": "The plant is healthy with no visible signs of disease.",
        "prevention": "Continue good gardening practices, including proper watering and pest control."
    },
    4: {
        "name": "Ring Spot",
        "details": "Ring Spot shows up as circular yellow or brown lesions on leaves, which can merge and cause defoliation.",
        "prevention": "Remove infected plants, use virus-free seeds, and control aphid populations."
    }
}

# Assuming you have already trained the model and loaded it
# Example of loading a pre-trained model (modify the path as needed)
model = tf.keras.models.load_model('/content/drive/MyDrive/plant_disease_model.h5')

# Example of loading an image for prediction (modify the image path)
image_path = '/content/Screenshot 2024-10-13 003406.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))  # Resize to match the input size of the model
image = image / 255.0  # Normalize the image

# Predict the disease from the image
predictions = np.argmax(model.predict(np.expand_dims(image, axis=0)), axis=1)
predicted_class = predictions[0]  # Get the predicted class (0 to 4)

# Fetch the corresponding disease information
disease_name = disease_info[predicted_class]['name']
disease_details = disease_info[predicted_class]['details']
disease_prevention = disease_info[predicted_class]['prevention']

# Display the output
print(f"Disease: {disease_name}")
print(f"Details: {disease_details}")
print(f"Prevention: {disease_prevention}")
