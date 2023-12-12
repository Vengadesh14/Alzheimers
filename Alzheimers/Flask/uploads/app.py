import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow import keras
import pandas as pd
import cv2
import tensorflow as tf
#flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from keras.models import load_model

global graph

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()

graph=tf.compat.v1.get_default_graph()

app = Flask(__name__)
set_session(sess)

@app.route('/',methods=['GET'])
def index():
    return render_template('alzheimers.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        #Get the file from post request
        f = request.files['image']
        #save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(180, 180))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data=preprocess_input(x)
        print(img_data.shape)
        #img_data.shape
        #model._make_predict_function() 
        
        with graph.as_default():
            model = load_model('adp.h5')
            output=np.argmax(model.predict(img_data), axis=1)
        
        if output == 0:
            text = "Mild Demented"
        elif output == 1:
            text = "Moderate Demented"
        elif output == 2:
            text = "Non Demented"
        else:
            text = "Very Mild Demented"
        
        return text
        
    return '1'

if __name__ == "__main__":
    app.run(debug=True)
