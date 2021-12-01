from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import base64
import io
from io import BytesIO



app = Flask(__name__)
model = keras.models.load_model('model2.h5') #loading the model


input_list = ["Aquarius", "Aries", "Cancer", "Canis Major", "Cassiopeia", "Cygnus", "Leo", "Lyra", "Orion", "Pisces", "Scorpius", "Taurus", "Ursa Minor", "Virgo"]
@app.route("/")
def index():
    return render_template("index.html")


@app.route('/hook', methods=['POST'])
def get_image():
    content = request.form.get("imageBase64").split(';')[1]
    img_enc = content.split(',')[1]
    body = base64.decodebytes(img_enc.encode('utf-8'))
    img = Image.open(BytesIO(body)).convert('L') # L means turn gray
    img = img.resize((128,128))
    img= np.asarray(img)
    img = img/255
    img = img.reshape(1,128,128,1)
    prediction = input_list[np.argmax(model.predict(img))]
    my_dict = {'pred': prediction}
    return my_dict
