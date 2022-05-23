# app.py
from importlib.resources import path
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import modeltostring

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
filename = ''
data_text = ''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return {
        "app_name": "Receipt Recognition",
        "status": "normal"
    }


@app.route('/scan', methods=['POST'])
def upload_image():
    file = request.files['image']
    print(file.filename)

    if file.filename == '':
        return {
            'error': 'No image selected for uploading'
        }

    if file and allowed_file(file.filename):
        # Upload
        filename = secure_filename(file.filename)

        file.save(app.config['UPLOAD_FOLDER']+filename)

        print('upload_image filename: ' + filename)

        # reading image
        img = cv2.imread(app.config['UPLOAD_FOLDER']+filename)
        data_text = modeltostring.img_to_string(img)

        return {
            "result": data_text.split(' ')
        }
    else:
        return {
            'error': 'Allowed image types are - png, jpg, jpeg, gif'
        }


if __name__ == "__main__":
    app.run(debug=True)
