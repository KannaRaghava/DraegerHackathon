import os
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from predict import prediction

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_file():
    filename = ""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            index = prediction(UPLOAD_FOLDER + filename).flatten().argmax()
            masks_dict = { 0: "3M Blue", 1: "3M Grey", 2: "Draeger Black", 3: "Draeger Blue"}
            print(index)
            return masks_dict[index]
            # return redirect(url_for('uploaded_file', filename=filename))
    return "<html><body><p>404</p></body></html>"


if __name__ == '__main__':
    app.secret_key = "SomeRandomString"
    app.run()
