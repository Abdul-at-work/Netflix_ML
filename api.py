# Dependencies
from typing_extensions import runtime
from flask import Flask, request, jsonify
from flask import url_for, redirect, render_template
import joblib
import traceback
import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
import os

# Your API definition
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("forest.html")

@app.route('/yes')
def yes():
    return render_template("yes.html")
    

@app.route('/no/<genre>/<runtime>/<language>/<year>/<score>/')        
def no(genre,runtime,language,year,score):
    if int(score)==1:
        x=0
    else:
        x=1
    final=[genre,runtime,language,year,x]
    traindata = pd.DataFrame([final],columns=["Genre","Runtime","Language","Year","IMDB Score"])     
    traindata.to_csv('train/data.csv',mode='a',index=False,header=not os.path.exists('train/data.csv'))
    traindata.to_csv('train/train.csv',mode='a',index=False,header=False)
    return render_template('no.html') 
    


@app.route('/predict', methods=['POST','GET'])
def predict():
    if neigh:
        try:
            final=[]
            for x in request.form.values():
                print(x)
                final.append(x)
            query = pd.DataFrame([final],columns=["Genre","Runtime","Language","Year"])
            le = preprocessing.LabelEncoder()
            netflix=pd.read_csv("train/train.csv")
            enc=le.fit(netflix["Language"])
            query["Language"]=enc.transform(query["Language"])
            enc1=le.fit(netflix["Genre"])
            query["Genre"]=enc1.transform(query["Genre"])
            print(query)
            prediction = neigh.predict(query)
            print(prediction) 
            df_list=query.values.tolist()
            df_list=df_list[0]
            print(df_list)
            if int(prediction)==1:
                return render_template('netflix.html',pred='The movie is a Success',genre=final[0],runtime=final[1],language=final[2],year=final[3],score=int(prediction))
            else:
                return render_template('netflix.html',pred='The movie is a Flop',genre=final[0],runtime=final[1],language=final[2],year=final[3],score=int(prediction))
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    neigh = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)