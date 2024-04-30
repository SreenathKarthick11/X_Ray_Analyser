# Initialising a varaible to store the path to dataset newXray
# newXray contains 9 classes : fractured and normal xrays of arm, hand, hindlimb and foot along with Error Images
dataset_path = r"C:\Users\HP\Desktop\Xrays\newXray" 

# Importing Necessary Libraries
import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Defining a function image_preprocessing that standardises various parameters of each picture.
def image_preprocessing(path):

    # Initialising two lists to store images and corresponding labels
    images = [] 
    labels = []

    # This loop iterates through each class within the directory stored at the location in base_path
    for class_name in os.listdir(path):
        # class_path now contains path to each class within the directory
        class_path = os.path.join(path, class_name)

        # Handling exception, i.e. if a particular file within the dataset is not a directory, skip it.
        if not os.path.isdir(class_path):
            continue 
    
        # This loop iterates through each image in class_path
        for image in os.listdir(class_path):

            # Storing the path to each image in image_path
            image_path = os.path.join(class_path, image)

            # Try Except Block to avoid cases where image is missing or any other error in reading the image
            try:
                # cv2.imread reads every image and also converts it to grayscale
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Histogram equalisation is performed to improve contrast of the picture thus enabling model to perform better
                img_eq = cv2.equalizeHist(img)

                # Resizing the images so that every image is of a certain fixed dimension, here 224x224
                img = cv2.resize(img_eq, (224, 224))

                # The datatype of everyvalue in img is converted to float and each pixel value is divided by 255 inorder to convert values to the range [0,1]
                img = img.astype('float32') / 255.0

                # To handle cases where image at a path could not be read. If so, skip it
                if img is None:
                    print(f"Error: Could not read image: {image_path}")
                    continue  

                # flatten converts the 2D array corresponding to each pre_processed image to a 1D array
                # Thereafter each array is appended to the list images. Similarly class_names are appended to labels list
                images.append(img.flatten())
                labels.append(class_name)

            # In case of IOError (where file couldn't be opened or unexpected error during opening) or OSError (anyother error mainly system errors), skip to next image
            except (IOError, OSError) as e:
                print(f"Error processing image: {image_path} ({e})")
                continue  # Skip to next image

    # The function returns a tuple containing two numpy arrays : images (containing pictures with modified parameters) 
    # and labels (indicating if the image contains fractured or normal xray and body part involved.)
    return np.array(images), np.array(labels)

# Function call, assigning images and labels to variables X and y
X, y = image_preprocessing(dataset_path)

# Using train_test_split to split dataset to training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# StandardScaler standardises values corresponding to each parameters
# Using Decision Tree Classifier 
knn = KNeighborsClassifier()

# Training the SVM Model pipeline
knn.fit(X_train, y_train)

# Predicting test data to evaluate performance
y_pred = knn.predict(X_test)

# Returns the accuracy of the model
knn.score(X_test,y_test)

# Classification Report shows precision, recall, f1-score and support in case of each class
print(classification_report(y_test, y_pred))
