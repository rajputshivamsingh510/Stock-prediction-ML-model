from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load or train models
def get_or_train_model(ticker, start_date, end_date=None, model_type="random_forest"):
    # Download data
    dataset = yf.download(ticker, start=start_date, end=end_date)
    
    # Preprocess data (similar to your existing code)
    dataset.columns = dataset.columns.get_level_values(0)
    
    # Scale the data
    ms = MinMaxScaler()
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    ms.fit(dataset[features])
    scaled_data = ms.transform(dataset[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=dataset.index)
    
    # Define x and y
    x = scaled_df.drop(columns=["Close"])
    y = scaled_df["Close"]
    
    # Train model based on type
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x, y)
        
        # Make predictions
        y_pred = model.predict(x)
        
    elif model_type == "lstm":
        # Prepare data for LSTM
        scaled = ms.fit_transform(dataset[['Close']])
        
        x_lstm = []
        y_lstm = []
        
        for i in range(60, len(scaled)):
            x_lstm.append(scaled[i-60:i, 0])
            y_lstm.append(scaled[i, 0])
            
        x_lstm = np.array(x_lstm)
        y_lstm = np.array(y_lstm)
        
        x_lstm = np.reshape(x_lstm, (x_lstm.shape[0], x_lstm.shape[1], 1))
        
        # Build and train LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(60, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_lstm, y_lstm, epochs=5, batch_size=32)
        
        # Make predictions
        y_pred = model.predict(x_lstm).flatten()
        
        # Adjust y and dataset for metrics calculation
        y = y[-len(y_pred):]
        dataset = dataset.iloc[-len(y_pred):]
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Prepare results
    actual_prices = dataset['Close'].values
    
    # For LSTM, the predictions need to be inverse transformed
    if model_type == "lstm":
        pred_array = np.array(y_pred).reshape(-1, 1)
        temp_array = np.zeros((len(pred_array), len(features)))
        temp_array[:, features.index("Close")] = pred_array[:, 0]
        predicted_prices = ms.inverse_transform(temp_array)[:, features.index("Close")]
    else:
        temp_array = x.copy()
        temp_array['Close'] = y_pred
        inverse_scaled = ms.inverse_transform(temp_array)
        predicted_prices = inverse_scaled[:, features.index("Close")]
    
    # Create result dictionary
    result = {
        "ticker": ticker,
        "model": model_type,
        "metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        },
        "data": []
    }
    
    # Add data points (last 30 or less)
    num_points = min(30, len(actual_prices))
    for i in range(num_points):
        idx = -num_points + i
        result["data"].append({
            "date": dataset.index[idx].strftime('%Y-%m-%d'),
            "actual": float(actual_prices[idx]),
            "predicted": float(predicted_prices[idx])
        })
    
    return result

# Define routes
@app.route('/')
def home():
    return render_template('index.html')  # Serve your HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        ticker = request.form['ticker']
        start_date = request.form['start-date']
        end_date = request.form['end-date'] if request.form['end-date'] else None
        model_type = request.form['model']
        
        # Get predictions
        result = get_or_train_model(ticker, start_date, end_date, model_type)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)