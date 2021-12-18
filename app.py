from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('cats_dogs_class_model.h5')

class_dict = {0: 'Cats (Kucing)', 1: 'Dogs (Anjing)'}

def predict_label(img_path):
    query = cv2.imread(img_path)
    output = query.copy()
    query = cv2.resize(query, (32, 32))
    q = []
    q.append(query)
    q = np.array(q, dtype='float') / 255.0
    q_pred = model.predict(q)
    predicted_bit = int(q_pred)
    return class_dict[predicted_bit]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '_main_':
    app.run(debug=True)