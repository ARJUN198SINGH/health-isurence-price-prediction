import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

application = Flask(__name__)
app=application


model=pickle.load(open('models/healthmodel.pkl','rb'))
standard_scaler=pickle.load(open('models/healthstand.pkl','rb'))
# pca=pickle.load(open('models/pcapokemon.pkl','rb'))


@app.route('/')
def index():
    return 'hiii'

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        age=float(request.form.get('age'))
        bmi = float(request.form.get('bmi'))
        children = float(request.form.get('children'))
        region = float(request.form.get('region'))
        sex_female = float(request.form.get('sex_female'))
        sex_male = float(request.form.get('sex_male'))

        smoker_no = float(request.form.get('smoker_no'))
        smoker_yes = float(request.form.get('smoker_yes'))
        # Speed	 = float(request.form.get('Speed'))
        
        # Generation	 = float(request.form.get('Generation'))
    
        new_data_scaled=standard_scaler.transform([[age,bmi,children,region,sex_female,sex_male,smoker_no,smoker_yes]])
        # new_data_pca=pca.transform(new_data_scaled)
        result=model.predict(new_data_scaled)
        # if(result[0]==1):
        #     result='legendary'
        # else:
        #     result='not legendary'
        return render_template('insurence.html',result=result)
    else:  
        return render_template('insurence.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
