import cv2
import numpy as np
import joblib 
from sklearn.preprocessing import LabelEncoder

svm_model = joblib.load('image_classifier_svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


img_path = 'data/train/application/application_2.jpg'  
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


if img is None:
    print(f"Error: Unable to load image at {img_path}")
else:
    img_resized = cv2.resize(img, (200, 200))
    img_flattened = img_resized.flatten().reshape(1, -1)  

    
    predicted_class_idx = svm_model.predict(img_flattened)
    predicted_class = label_encoder.inverse_transform(predicted_class_idx)

    print(f"Predicted class: {predicted_class[0]}")
#run to0 check model 
#joblib to load vpkl model;
#checvk only for traimned data set as provided by me prathmesh
