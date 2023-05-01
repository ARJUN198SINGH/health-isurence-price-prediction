import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

application = Flask(__name__)
app=application


model=pickle.load(open('models/gscvpokemon.pkl','rb'))
standard_scaler=pickle.load(open('models/scalerpokemon.pkl','rb'))
pca=pickle.load(open('models/pcapokemon.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Type1=float(request.form.get('Type1'))
        Type2 = float(request.form.get('Type2'))
        Total = float(request.form.get('Total'))
        HP = float(request.form.get('HP'))
        Attack = float(request.form.get('Attack'))
        Defense = float(request.form.get('Defense'))

        SpAtk = float(request.form.get('SpAtk'))
        SpDef = float(request.form.get('SpDef'))
        Speed	 = float(request.form.get('Speed'))
        
        Generation	 = float(request.form.get('Generation'))
    
        new_data_scaled=standard_scaler.transform([[Type1,Type2, Total, HP, Attack, Defense, SpAtk,SpDef, Speed, Generation]])
        new_data_pca=pca.transform(new_data_scaled)
        result=model.predict(new_data_pca)
        if(result[0]==1):
            result='legendary'
        else:
            result='not legendary'
        return render_template('pokemon.html',result=result)
    else:  
        return render_template('pokemon.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
