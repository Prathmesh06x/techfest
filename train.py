import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib


def load_images_from_directory(directory, size=(200, 200)):
    images = []
    labels = []
    
    
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img_resized = cv2.resize(img, size)  
                    img_flattened = img_resized.flatten()  
                    images.append(img_flattened)
                    labels.append(label)
                else:
                    print(f"Warning: Image at {img_path} could not be loaded and will be skipped.")
    
    return np.array(images), np.array(labels)


train_dir = 'data/train'  
val_dir = 'data/val'      


X_train, y_train = load_images_from_directory(train_dir)


X_val, y_val = load_images_from_directory(val_dir)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

svm_model = SVC(kernel='linear', probability=True)  # Yes
svm_model.fit(X_train, y_train_encoded)

# Predy
y_pred = svm_model.predict(X_val)

# Evaluat
print("Classification Report:")
print(classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_))

# Savy
joblib.dump(svm_model, 'image_classifier_svm_model.pkl')

# joblin ytyo load pkl
joblib.dump(label_encoder, 'label_encoder.pkl')
