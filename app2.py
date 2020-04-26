# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:04:21 2020

@author: Admin
"""

from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd
import numpy as np

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method=='POST':
        message=request.form["message"]
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__=='__main__':
    app.run(debug=True)
    