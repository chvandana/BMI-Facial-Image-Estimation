from flask import Flask, render_template, request, redirect, url_for
import sklearn
import numpy as np
from PIL import Image
import joblib
import face_recognition
from werkzeug.utils import secure_filename
import os



app = Flask(__name__)
UPLOAD_FOLDER = 'static'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_face_encoding(image_path):

    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        return "no face found !!!"
        # print("no face found !!!")
        # return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()


def predict_height_weight_BMI(test_image, height_model, weight_model, bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)), axis=0)
    height = np.asscalar(np.exp(height_model.predict(test_array)))
    weight = np.asscalar(np.exp(weight_model.predict(test_array)))
    bmi = np.asscalar(np.exp(bmi_model.predict(test_array)))
    return height, weight, bmi


@app.route('/')
def home():
    return render_template('login.html')


@app.route('/index', methods=['POST', 'GET'])
def home1():
    return render_template('index.html')



@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        test_image = request.files['file']
        height_model = joblib.load('height.pkl')
        weight_model = joblib.load('weight.pkl')
        bmi_model = joblib.load('bmi.pkl')
        
        if get_face_encoding(test_image) == "no face found !!!":
            return render_template('result.html',filename=filename,height=0, weight=0, bmi=0)
        else:
            test_array = np.expand_dims(np.array(get_face_encoding(test_image)), axis=0)
            height = round(np.exp(height_model.predict(test_array)).item(), 3)
            weight = round(np.exp(weight_model.predict(test_array)).item(), 3)
            bmi = round(np.exp(bmi_model.predict(test_array)).item(), 3)
            return render_template('result.html',filename=filename,height=height, weight=weight, bmi=bmi)


if __name__ == '__main__':
    app.run()
