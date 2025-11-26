import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Import pipeline functions
from data.data_preprocessing import merge_and_preprocess_data
from .feature_engineer import add_technical_indicators, create_lstm_sequences, train_val_test_split_sequences

def build_lstm_model(input_shape):
    """Builds, compiles, and returns a Keras LSTM model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def run_training_pipeline(ticker="AAPL", fred_series_id="DGS10", start_date="2015-01-01",
                          seq_len=60, epochs=50, batch_size=32):
    """Runs the full LSTM pipeline: Data -> Features -> Scaling -> Sequencing -> Training -> Evaluation"""
    print(f"\n--- [1/6] Starting LSTM Pipeline for {ticker} ---")

    merged_df = merge_and_preprocess_data(ticker=ticker, fred_series_id=fred_series_id, start_date=start_date)
    features_df = add_technical_indicators(merged_df.copy(), price_col='Close')

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(features_df), index=features_df.index, columns=features_df.columns)

    TARGET = 'Close'
    FEATURES = [col for col in df_scaled.columns if col != TARGET]

    X, y = create_lstm_sequences(df_scaled, feature_cols=FEATURES, target_col=TARGET, seq_len=seq_len)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(X, y)

    print(f"\n--- [3/6] Building and Training Model ---")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0
    )

    print(f"\n--- [5/6] Evaluating Model ---")
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Test Loss (MSE): {loss:.6f}")

    model_path = f"models/lstm_model_{ticker}.h5"
    model.save(model_path)
    print(f"\n--- [6/6] Model Saved to {model_path} ---")

    return model, history

if __name__ == "__main__":
    run_training_pipeline(ticker="MSFT", epochs=5)