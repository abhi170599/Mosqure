import base64
import json
from io import BytesIO

import matplotlib.pyplot as plt
from Utils.utils import get_predictions,get_predictions_ahead

import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import pickle
import tensorflow as tf 

regressor = pickle.load(open('Utils/model_regr_1.pkl','rb'))

graph = tf.get_default_graph()


# from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#load the time series model
model = load_model('Utils/my_model_updated5.h5')
print(model.summary())


# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/get_predictions',methods=['POST'])
def get_prediction():

    map_points = request.json['map_points']

    pred = get_predictions(map_points,regressor)    

    json_response = {'pred' : pred}
    print(json_response)
    #print(json_response)
    return jsonify(json_response)                                        


@app.route('/get_predictions_future',methods=['POST'])
def get_prediction_in_future():

    map_points      = request.json['map_points']
    ahead          =  request.json['ahead']
    
    global graph
    with graph.as_default():
        pred = get_predictions_ahead(map_points,model,regressor,ahead)

    json_response = {'pred':pred}

    return jsonify(json_response)



if __name__=="__main__":
    app.run(host='0.0.0.0') 
