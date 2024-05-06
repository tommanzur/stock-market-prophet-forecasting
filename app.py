from flask import Flask, jsonify, request
import logging
import requests
from datetime import datetime
import pandas as pd
from prophet import Prophet
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = 'cf6313921d5ef0e4942ed6352948d3631051b44f6de6116dab33ffaaad9411bf'
headers = {'Authorization': 'Apikey ' + API_KEY}

def load_data(symbol):
    url = f'https://min-api.cryptocompare.com/data/v2/histoday'
    parameters = {
        'fsym': symbol,
        'tsym': 'USD',
        'limit': '2000',
        'toTs': int(datetime.now().timestamp())
    }
    response = requests.get(url, headers=headers, params=parameters)
    
    data = pd.DataFrame(response.json()['Data']['Data'])
    data['timestamp'] = pd.to_datetime(data['time'], unit='s')
    data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d')  # Formatear como cadena ISO 8601
    return data[['timestamp', 'close']]

@app.route('/api/data/<symbol>', methods=['GET'])
def get_data(symbol):
    data = load_data(symbol)
    logging.warning("DATA===================================)")
    logging.warning(data.to_dict(orient='records'))
    return data.to_dict(orient='records')

@app.route('/api/forecast/<symbol>', methods=['GET'])
def get_forecast(symbol):
    data = load_data(symbol)
    df_train = data.rename(columns={'timestamp': 'ds', 'close': 'y'})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S')  # Formatear como cadena ISO 8601
    logging.warning("FORECAST===================================)")
    logging.warning(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'))
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
