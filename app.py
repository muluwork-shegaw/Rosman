import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('22-08-2020-16-45-50-513.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict_test',methods=['POST'])
def predict_test():
    
    from_date  = request.values.get('from_date')
    to_date  = request.values.get('to_date')

    mydates = pd.date_range(from_date, to_date)
    dates_df = pd.DataFrame(mydates, columns = ["ds"])


    forcast = model.predict(dates_df)
    result = forcast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')

    dates = result.ds.to_list()
    prediction = result.yhat.to_list()


    prediction = [ round(elem, 2) for elem in prediction ]
    

    data = {
        "dates": dates,
        "prediction": prediction,      

    }  
    

    return data
 


if __name__ == "__main__":
    app.run(debug=True)