from flask import Flask, render_template, request
import pandas
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('tour_model02.pkl','rb'))

@app.route('/',methods = ['GET'])
def home():
    return render_template ('index.html')
    #http://127.0.0.1:5000

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        DurationOfPitch = float(request.form['DurationOfPitch'])
        Age = float(request.form['Age'])
        MonthlyIncome = float(request.form['MonthlyIncome'])
        PreferredPropertyStar = float(request.form['PreferredPropertyStar'])
        NumberOfTrips = float(request.form['NumberOfTrips'])
        Passport = float(request.form['Passport'])
        CityTier = float(request.form['CityTier'])
        
        values = np.array([[DurationOfPitch,Age,MonthlyIncome,PreferredPropertyStar,NumberOfTrips,Passport,CityTier]])
        prediction = model.predict(values)

        return render_template('result.html', prediction = prediction)
    

if  __name__ == "__main__":
    app.run(debug = True)
