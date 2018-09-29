import os

from flask import Flask, request
from werkzeug.utils import secure_filename

from predict import prediction

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return '<html><body><p>No image sent</p></body></html>'
        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return '<html><body><p>Empty image sent</p></body></html>'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            index = prediction(UPLOAD_FOLDER + filename).flatten().argmax()
            masks_dict = {0: '3M Blue', 1: '3M Grey', 2: 'Draeger Black', 3: 'Draeger Grey'}
            return masks_dict[index]
    return "<html><body><p>404</p></body></html>"


if __name__ == '__main__':
    app.secret_key = 'SomeRandomString'
    app.run(host='0.0.0.0', port=8888)
