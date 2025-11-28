"""LSTM model training pipeline.

Summary of recent feature changes:
- Uses `add_technical_indicators` from `feature_engineer` to include engineered features.
- Target is defined as `Close_Future` shifted by `predict_ahead` days to support flexible
    forecasting horizons.
- Separate scalers are used for features and the target to preserve proper inverse-scaling
    when evaluating predictions.
- Saves trained model to `models/` and returns `target_scaler` for downstream use.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Import pipeline functions
from src.data.data_preprocessing import merge_and_preprocess_data
from src.ai.feature_engineer import add_technical_indicators, create_lstm_sequences, train_val_test_split_sequences

def build_lstm_model(input_shape):
    """Builds, compiles, and returns a Keras LSTM model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1) # Output is a single predicted value (the future price)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_training_pipeline(ticker="AAPL", fred_series_id="DGS10", start_date="2015-01-01",
                          seq_len=60, epochs=50, batch_size=32, predict_ahead=1):
    """Runs the full LSTM pipeline: Data -> Features -> Scaling -> Sequencing -> Training -> Evaluation."""
    print(f"\n--- [1/6] Starting LSTM Pipeline for {ticker} ---")

    merged_df = merge_and_preprocess_data(ticker=ticker, fred_series_id=fred_series_id, start_date=start_date)
    features_df = add_technical_indicators(merged_df.copy(), price_col='Close')
    
    # --- FIXED: Target Definition and Scaling ---
    
    # Define Target: The price N days in the future
    TARGET_COL = 'Close_Future'
    features_df[TARGET_COL] = features_df['Close'].shift(-predict_ahead)
    features_df.dropna(inplace=True) # Drop the last row with NaN target
    
    # Features for LSTM (only raw prices/volume for pure temporal analysis)
    # This list ensures only the features the LSTM should see are scaled together
    LSTM_FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Create separate scalers for Features (X) and Target (y)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform features
    features_df[LSTM_FEATURE_COLS] = feature_scaler.fit_transform(features_df[LSTM_FEATURE_COLS])
    
    # Fit and transform the target
    features_df[TARGET_COL] = target_scaler.fit_transform(features_df[[TARGET_COL]])
    
    print(f"\n--- [2/6] Data Sequencing ---")
    X, y = create_lstm_sequences(
        df=features_df, 
        feature_cols=LSTM_FEATURE_COLS, 
        target_col=TARGET_COL, 
        seq_len=seq_len
    )

    print(f"X shape: {X.shape} (samples, timesteps, features), y shape: {y.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(X, y)

    print(f"\n--- [3/6] Building and Training Model ---")
    # Input shape: (timesteps, features)
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1 # Set verbose to 1 to see progress
    )

    print(f"\n--- [4/6] Evaluating Model ---")
    loss = model.evaluate(X_test, y_test, verbose=0)
    
    # Evaluation requires inverse transforming predictions for a meaningful RMSE
    y_pred_scaled = model.predict(X_test)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    
    print(f"LSTM Test Loss (Scaled MSE): {loss:.6f}")
    print(f"LSTM Test RMSE (Original Price Scale): ${rmse:.4f}")

    # Create model directory if it doesn't exist
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model_path = f"models/lstm_model_{ticker}.h5"
    model.save(model_path)
    print(f"\n--- [5/6] Model Saved to {model_path} ---")

    # Pass the target_scaler back for future use (e.g., XGBoost integration or final deployment)
    return model, history, target_scaler 

if __name__ == "__main__":
    model, history, target_scaler = run_training_pipeline(ticker="MSFT", epochs=5)
    print("\nLSTM Training Pipeline finished successfully.")