#!/usr/bin/env python
# coding: utf-8

# In[8]:


from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os
import pickle
import pandas as pd


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)    
    loaded_model = pickle.load(open("emisi_flask.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result', methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list_values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result)== 1:
            prediction = 'Intensitas Emisi lebih besar dari 256,93'
        else:
            prediction = 'Intensitas Emisi lebih kecil dari 256,93'
        return render_template('result.html',prediction == prediction)
    


if __name__ == '__main__':
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] =True
    

