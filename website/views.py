# views.py file for backend fuction like image processing and prediction with svm model
# Import all previous modules
from flask import Blueprint, Flask, render_template, request, flash,redirect,url_for
from flask_login import login_required, current_user
from . import db
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
from werkzeug.utils import secure_filename
import os
# OpenCV is used for image processing like canny edge tecnique
# PIL is used to save the image file for reuseability
# import sklearn for svm model
# trained model is saved in a file using pickle. This allows you to persist the model to disk,
# and later load it back into memory to make predictions on new data without having to retrain the model from scratch.
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler  # Assuming you used StandardScaler
import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from keras.layers import TFSMLayer
####
# Load the saved model

####

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

views = Blueprint('views', __name__) # making blueprint for views.py
# intialising values
file_name=""  # intial filename
lt=50         # instial value of lower threshold
ut=80         # instial value of upper threshold
file_name1="" # intial name of canny image file
@views.route('/', methods=['GET', 'POST'])  # routing home page 
@login_required
def home():
    global lt,ut
    global file_name
    global file_name1
    global Predic_class
    form = UploadFileForm()
    
    Predic_class=""  # intailising the output prediction from model
    if 'type' in request.form: # if values are entered in change threshold form 
        re=request.form["type"]
        if re=="request1":
            lt=int(request.form["LowerT"]) # get the value of lower threshold from the form
            ut=int(request.form["UpperT"]) # get the value of upper threshold from the form
        file_name1=canny(file_name,lt,ut)  # calling the canny function reset the lower threshold, upper threshold value
        return render_template("home.html", user=current_user,form=form,file_name=file_name,file_name1=file_name1, lt=lt, ut=ut,Predic_class=Predic_class) # render the home template
    if 'p' in request.form:  # if predict button is pressed 
        if request.form["pred"]!=None:  # if the pred textbox is empty
            model_pkl_file = r"C:\Users\hp\Desktop\xray_web\website\static\SVM_Model.pkl"  # calling the model
            with open(model_pkl_file, 'rb') as file:
                loaded_pipe = pickle.load(file)

            # getting the image path
            img_path = r"C:\Users\hp\Desktop\xray_web\website\static" 
            img_path=os.path.join(img_path,file_name)

            # Load the new image
            img = cv2.imread(img_path)

            # Preprocess the new image (assuming grayscale conversion and resizing were used during training)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if applicable
            resized = cv2.resize(gray, (224, 224))  # Resize to match training size (adjust if different)
            normalized = resized.astype('float32') / 255.0  # Normalize pixel values

            # Flatten the image (assuming this was done during training)
            new_image_data = normalized.flatten()

            # Reshape if necessary (based on how training data was flattened)
            # For example, if X_train.shape was (n_samples, features), reshape to (1, features)
            new_image_data = new_image_data.reshape(1, -1)  # Reshape to a 2D array with 1 sample

            # Make prediction using the loaded model
            prediction = loaded_pipe.predict(new_image_data)


            # loaded_model = tf.keras.models.load_model('C:/Users/hp/Desktop/xray_web/website/static/Model2/Model2.h5')

            # # Load and preprocess the image
            # img_path = r"C:\Users\hp\Desktop\xray_web\website\static"
            # img_path=os.path.join(img_path,file_name)
            # img = image.load_img(img_path, target_size=(150, 150))  # Resize to match model input size
            # img_array = image.img_to_array(img)
            # img_array = np.expand_dims(img_array, axis=0)
            # img_array = preprocess_input(img_array)

            # # Predict
            # predictions = loaded_model.predict(img_array)

            # # Interpret the predictions
            # class_indices = {'Fractured_Arm': 0, 'Fractured_Foot': 1, 'Fractured_Hand': 2, 'Fractured_Hindlimb': 3, 'Normal_Arm': 4, 'Normal_Foot': 5, 'Normal_Hand': 6, 'Normal_Hindlimb': 7}
            # predicted_class = list(class_indices.keys())[np.argmax(predictions)]
            Predic_class=prediction[0]
            return render_template("home.html", user=current_user,form=form,file_name=file_name,file_name1=file_name1, lt=lt, ut=ut,Predic_class=Predic_class)  # render the home template
        else:
            Predic_class=""
        

    if form.validate_on_submit(): # get the file from submit button
        file = form.file.data  # Grab the uploaded file
        # Validate filename for security
        
        file_name = secure_filename(file.filename)
        # save the file
        file.save(os.path.join(r"C:\Users\hp\Desktop\xray_web\website\static", file_name))

        file_name1=canny(file_name,lt,ut)
        
        # Optional: Handle success, e.g., redirect or flash a message

        # #####
        

    return render_template("home.html", user=current_user,form=form,file_name=file_name,file_name1=file_name1, lt=lt, ut=ut,Predic_class=Predic_class)

# canny function
def canny(file_name,lt,ut): 
    # Read the image using OpenCV
    input_image = cv2.imread(os.path.join(r"C:\Users\hp\Desktop\xray_web\website\static", file_name), cv2.IMREAD_GRAYSCALE)
    file_name1= file_name[:-4] + "out.png"    
    # Perform Canny edge detection
    if lt!=50 and ut!=80:
        if type(lt)==int and type(ut)==int:    
            edges = cv2.Canny(input_image, threshold1=lt, threshold2=ut)  # Adjust thresholds as needed
        else:
            flash('Theshold should be numbers', category='error')
            edges = cv2.Canny(input_image, threshold1=50, threshold2=80)  # Adjust thresholds as needed
    else:
        edges = cv2.Canny(input_image, threshold1=50, threshold2=80)  # Adjust thresholds as needed
    # Check if edges is not None before converting to PIL image
    if edges is not None:
        # Convert the Canny image array to PIL format
        pil_image = Image.fromarray(edges)

        # Save or display the PIL image
        pil_image.save(os.path.join(r"C:\Users\hp\Desktop\xray_web\website\static", file_name1))  # Save the PIL image
        # pil_image.show()  # Display the PIL image
        return file_name1
    else:
        print("Error: Canny edge detection failed.")