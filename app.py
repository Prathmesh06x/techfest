import os
from flask import Flask, request, render_template, redirect, url_for
import joblib  
from PIL import Image
import numpy as np

model = joblib.load('C:\\Users\\ASUS\\Music\\iitbombay\\image_classifier_svm_model.pkl')

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'  
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

# labelsx
class_labels = ['application', 'email', 'letter', 'questionnaire']


def classify_document(image_path):
    
    img = Image.open(image_path).convert('L') 
    img_resized = img.resize((200, 200))  
    img_array = np.array(img_resized) / 255.0  #normlzing
    img_flattened = img_array.flatten().reshape(1, -1)  
    
    # Get prediction from the model
    prediction = model.predict(img_flattened)
    class_idx = prediction[0]  
    return class_labels[class_idx], 1.0 


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            
            try:
                result, confidence = classify_document(file_path)
            except ValueError as e:
                return f"Error: {str(e)}", 400

            return render_template('RESULT.HTML', result=result, confidence=confidence, filename=file.filename)
    return render_template('UPLOAD.HTML')


@app.route('/result')
def result():
    return render_template('RESULT.HTML')

if __name__ == '__main__':
    app.run(debug=True)
