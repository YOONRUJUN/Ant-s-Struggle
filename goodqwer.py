import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

def get_naver_stock_data(ticker, years):
    days = int(years) * 365
    url = f'https://fchart.stock.naver.com/sise.nhn?symbol={ticker}&timeframe=day&count={days}&requestType=0'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    items = soup.find_all('item')
    data = []
    for item in items:
        fields = item['data'].split('|')
        date = datetime.datetime.strptime(fields[0], '%Y%m%d')
        open_price = float(fields[1])
        high_price = float(fields[2])
        low_price = float(fields[3])
        close_price = float(fields[4])
        volume = int(fields[5])
        data.append([date, open_price, high_price, low_price, close_price, volume])
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.set_index('Date', inplace=True)
    return df

def predict_stock_prices(ticker, years):
    data = get_naver_stock_data(ticker, years)
    data = data.dropna()
    data['SMA'] = data['Close'].rolling(window=5).mean()
    data['min'] = data['Low'].rolling(window=5).min()
    data['max'] = data['High'].rolling(window=5).max()
    data['Range'] = data['High'] - data['Low']
    data.dropna(inplace=True)

    features = data[['min', 'max', 'SMA']]
    targets = data[['High', 'Low']]

    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)

    target_scaler = MinMaxScaler()
    scaled_targets = target_scaler.fit_transform(targets)

    X_train, X_test, y_train, y_test = train_test_split(scaled_features, scaled_targets, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model.fit(X_train, y_train)

    future_days = 5
    start_date = datetime.date.today()
    future_dates = pd.bdate_range(start=start_date, periods=future_days + 1)[1:]

    future_predictions = []
    current_features = scaled_features[-1].reshape(1, -1)

    for _ in range(future_days):
        next_day_prediction = model.predict(current_features)
        future_predictions.append(next_day_prediction[0])

        next_day_prediction_original = target_scaler.inverse_transform([next_day_prediction[0]])
        new_min = min(feature_scaler.inverse_transform(current_features)[0][0], next_day_prediction_original[0][1])
        new_max = max(feature_scaler.inverse_transform(current_features)[0][1], next_day_prediction_original[0][0])
        new_sma = (feature_scaler.inverse_transform(current_features)[0][2] * 4 + next_day_prediction_original[0][0]) / 5

        new_features = np.array([[new_min, new_max, new_sma]])
        current_features = feature_scaler.transform(new_features)

    future_predictions = target_scaler.inverse_transform(future_predictions)
    future_df = pd.DataFrame(future_predictions, columns=['High', 'Low'], index=future_dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['High'], mode='lines', name='High Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Low'], mode='lines', name='Low Price'))
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['High'], mode='lines', name='Future Predicted High', line=dict(dash='dash', color='orange')))
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Low'], mode='lines', name='Future Predicted Low', line=dict(dash='dash', color='blue')))
    fig.update_layout(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Price')

    return fig

st.title("Stock Price Predictor")
ticker = st.text_input("Enter the ticker symbol:", "005930")
years = st.number_input("Enter the number of years of data to fetch (e.g., 10 years of data):", 1, 20, 10)

if st.button("Predict Prices"):
    try:
        fig = predict_stock_prices(ticker, years)
        st.plotly_chart(fig)
    except Exception as e:
        st.error("Error: " + str(e))

