# Save model in .keras format
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
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
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
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Added EarlyStopping here
]

# Step 9: Train the CNN model
cnn_history = cnn_model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,  # Increased number of epochs
    validation_data=(X_val, y_val),
    callbacks=callbacks  # Add EarlyStopping here
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


cnn_model.save('/content/plant_disease_model.keras')
# prompt: /content/plant_disease_model.keras save in drive

# Mount Google Drive
# Copy the model file to your Google Drive
!cp /content/plant_disease_model.keras /content/drive/MyDrive/plant_disease_model.h5
import numpy as np
import cv2
import tensorflow as tf
from google.colab.patches import cv2_imshow # Import cv2_imshow


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

# Example of loading an image for prediction (modify the image path)
image_path = '/content/Screenshot 2024-10-13 004401.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))  # Resize to match the input size of the model
image = image / 255.0  # Normalize the image
#Show the picture
cv2_imshow(image) # Use cv2_imshow instead of cv2.imshow
# cv2.waitKey(0) # Remove this line as it's not needed with cv2_imshow
# cv2.destroyAllWindows() # Remove this line as it's not needed with cv2_imshow
# Predict the disease from the image
predictions = np.argmax(cnn_model.predict(np.expand_dims(image, axis=0)), axis=1)
predicted_class = predictions[0]  # Get the predicted class (0 to 4)

# Fetch the corresponding disease information
disease_name = disease_info[predicted_class]['name']
disease_details = disease_info[predicted_class]['details']
disease_prevention = disease_info[predicted_class]['prevention']

# Display the output
print(f"Disease: {disease_name}")
print(f"Details: {disease_details}")
print(f"Prevention: {disease_prevention}")


image_path = '/content/Screenshot 2024-10-13 154144.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))  # Resize to match the input size of the model
image = image / 255.0  # Normalize the image
#Show the picture
cv2_imshow(image) # Use cv2_imshow instead of cv2.imshow
# cv2.waitKey(0) # Remove this line as it's not needed with cv2_imshow
# cv2.destroyAllWindows() # Remove this line as it's not needed with cv2_imshow
# Predict the disease from the image
predictions = np.argmax(cnn_model.predict(np.expand_dims(image, axis=0)), axis=1)
predicted_class = predictions[0]  # Get the predicted class (0 to 4)

# Fetch the corresponding disease information
disease_name = disease_info[predicted_class]['name']
disease_details = disease_info[predicted_class]['details']
disease_prevention = disease_info[predicted_class]['prevention']

# Display the output
print(f"Disease: {disease_name}")
print(f"Details: {disease_details}")
print(f"Prevention: {disease_prevention}")
