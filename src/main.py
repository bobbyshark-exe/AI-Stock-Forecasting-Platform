import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import your modules
from src.data.data_preprocessing import merge_and_preprocess_data
from src.ai.feature_engineer import add_technical_indicators, create_lstm_sequences
from src.ai.model_trainer import run_training_pipeline
from src.ai.xgboost_trainer import build_xgboost_model

def run_hybrid_pipeline(ticker="MSFT", predict_ahead=1):
    print(f"--- STARTING HYBRID PIPELINE FOR {ticker} ---")
    
    # -------------------------------------------------------
    # PHASE 1: Train LSTM & Get Momentum Feature
    # -------------------------------------------------------
    print("\n>>> PHASE 1: Training LSTM Model...")
    
    # 1. Train the LSTM and get the model + scalers
    # We use a smaller epoch count for testing (e.g., 5 or 10)
    lstm_model, history, target_scaler = run_training_pipeline(
        ticker=ticker,
        # Increased epochs for better LSTM training (data-science mode)
        epochs=50,
        predict_ahead=predict_ahead
    )
    
    print("\n>>> PHASE 2: Generating LSTM Features (Momentum)...")
    
    # 2. Reload Data for Hybrid Integration
    # We need the full dataset again to generate predictions for XGBoost
    raw_df = merge_and_preprocess_data(ticker=ticker)
    df = add_technical_indicators(raw_df)
    
    # Define features matchings the LSTM training
    LSTM_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Scale Data (Must match the scaler logic in model_trainer.py)
    # Note: Ideally, we would pickle/save the exact scaler from Phase 1 to reuse here.
    # For this v1 implementation, we refit a new scaler on the same data range.
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[LSTM_COLS] = feature_scaler.fit_transform(df[LSTM_COLS])
    
    # Create Sequences for the WHOLE dataset
    SEQ_LEN = 60
    # We create a dummy target col just to satisfy the function signature
    df_scaled['dummy_target'] = 0 
    
    X_all, _ = create_lstm_sequences(
        df_scaled, 
        feature_cols=LSTM_COLS, 
        target_col='dummy_target', 
        seq_len=SEQ_LEN
    )
    
    # Predict with LSTM
    # This gives us the "Time-Series Momentum" score for every valid sequence
    lstm_preds_scaled = lstm_model.predict(X_all)
    lstm_preds = target_scaler.inverse_transform(lstm_preds_scaled)
    
    # -------------------------------------------------------
    # PHASE 3: Prepare Data for XGBoost (Hybrid)
    # -------------------------------------------------------
    print("\n>>> PHASE 3: Training Hybrid XGBoost...")
    
    # We need to align the LSTM predictions with the original DataFrame.
    # Since sequences consume SEQ_LEN rows, the predictions start at index SEQ_LEN.
    
    # Trim the original DF to match the LSTM output size
    hybrid_df = df.iloc[SEQ_LEN:].copy()
    hybrid_df['LSTM_Momentum'] = lstm_preds.flatten()
    
    # Define Target (Shifted Future Price)
    TARGET_COL = 'Close_Future'
    hybrid_df[TARGET_COL] = hybrid_df['Close'].shift(-predict_ahead)
    hybrid_df.dropna(inplace=True)
    
    # Define Features for XGBoost
    # Now we include the new 'LSTM_Momentum' feature!
    # We exclude ALL raw price columns to force reliance on LSTM and technical indicators
    EXCLUDE = ['Close', 'Open', 'High', 'Low', 'Volume', TARGET_COL, 'dummy_target']
    HYBRID_FEATURES = [c for c in hybrid_df.columns if c not in EXCLUDE]
    
    print(f"Hybrid Features: {HYBRID_FEATURES}")
    
    X_hybrid = hybrid_df[HYBRID_FEATURES]
    y_hybrid = hybrid_df[TARGET_COL]
    
    # Scale Hybrid Features
    scaler_xgb = StandardScaler()
    X_hybrid_scaled = scaler_xgb.fit_transform(X_hybrid)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid_scaled, y_hybrid, test_size=0.2, shuffle=False
    )
    
    # Train XGBoost
    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train, y_train)
    
    # Evaluation
    preds = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"\n>>> FINAL RESULT: Hybrid Model RMSE: ${rmse:.4f}")
    
    # Feature Importance
    importance = pd.Series(xgb_model.feature_importances_, index=HYBRID_FEATURES).sort_values(ascending=False)
    print("\n--- Top 10 Feature Importances (Hybrid) ---")
    print(importance.head(10))
    
    return xgb_model, importance

if __name__ == "__main__":
    run_hybrid_pipeline(ticker="MSFT")