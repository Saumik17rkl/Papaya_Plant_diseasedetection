Plant Disease Detection Using CNN

This project focuses on detecting plant diseases from images of leaves using a Convolutional Neural Network (CNN) model. The model is trained to classify different types of diseases, including Anthracnose, Bacterial Spot, Curl, Healthy, and Ring Spot. The data is preprocessed, augmented, and used to train the CNN model for accurate classification.

Features
Image Preprocessing: Resize, normalize, and prepare images from a custom dataset.
Data Augmentation: Uses ImageDataGenerator to apply random transformations to increase data variability.
CNN Model: Multi-layer Convolutional Neural Network using TensorFlow/Keras.
Early Stopping: Added to prevent overfitting and ensure optimal model performance.
Visualization: Accuracy and loss graphs, and a confusion matrix to analyze performance.
Evaluation: Displays classification report and metrics for model evaluation.
Dataset
The dataset contains images of papaya leaves labeled with the following disease classes:

Anthracnose
Bacterial Spot
Curl
Healthy
Ring Spot
Images are preprocessed and normalized for training.

Libraries Used
TensorFlow
NumPy
OpenCV
Scikit-learn
Matplotlib
Seaborn
tqdm (for progress bars)
Model Architecture
Input Layer: 128x128 RGB image.
Convolution Layers: Multiple Conv2D layers with BatchNormalization and MaxPooling for feature extraction.
Fully Connected Layer: Flattened feature maps connected to dense layers.
Dropout Layer: Regularization with dropout.
Output Layer: Softmax layer for multi-class classification.
Training
The model is trained using an augmented dataset with the following callbacks:

ReduceLROnPlateau: Reduces learning rate when a plateau in validation loss is detected.
EarlyStopping: Stops training if validation loss doesn’t improve, and restores the best model.
Instructions
Dataset Extraction:

The dataset is stored in a zip file and extracted using Python's zipfile module.
python
Copy code
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)
Data Loading and Preprocessing:

Images are loaded, resized, and normalized.
python
Copy code
img = cv2.imread(img_path)
img = cv2.resize(img, image_size)
Training the Model:

CNN model is compiled and trained with early stopping.
python
Copy code
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Model Saving:

Save the trained model in .h5 format.
python
Copy code
cnn_model.save('plant_disease_model.h5')
Evaluation:

The model is evaluated on the test set, and accuracy/loss curves are plotted.
python
Copy code
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
Confusion Matrix and Classification Report:

Visualize the performance with a confusion matrix and detailed classification report.
python
Copy code
sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Blues")
Installation
Clone the repository.

Install the required libraries by running:

bash
Copy code
pip install -r requirements.txt
Run the training script:

bash
Copy code
python train.py
Example Results

Future Improvements
Add more plant diseases to the dataset.
Explore different architectures for higher accuracy.
Deploy the model using FastAPI or Flask for real-time prediction.
