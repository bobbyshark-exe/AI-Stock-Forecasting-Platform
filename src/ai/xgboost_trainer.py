"""XGBoost training utilities for tabular forecasting.

Summary of recent feature changes:
- `prepare_tabular_data` now uses `add_technical_indicators` to include engineered
    indicators, shifts the target (`Close_Future`) by `predict_ahead`, and drops
    NaNs introduced by shifting.
- Uses `StandardScaler` for features (better baseline for tree models) and
    returns feature names and the fitted scaler for later interpretation.
- Trains XGBoost regressor and prints top feature importances.
"""

import numpy as np
import pandas as pd
# Import XGBoost for regression
try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    _XGBOOST_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import pipeline functions
from src.data.data_preprocessing import merge_and_preprocess_data
from src.ai.feature_engineer import add_technical_indicators # Use this to get engineered features

# --- DATA PREPARATION FOR XGBOOST ---

def prepare_tabular_data(ticker: str, fred_series_id: str, start_date: str, predict_ahead: int = 1) -> tuple:
    """
    Loads data, adds indicators, defines the target (shifted future price),
    and returns scaled tabular X and y arrays.
    """
    # 1. Data Ingestion and Feature Engineering
    merged_df = merge_and_preprocess_data(ticker=ticker, fred_series_id=fred_series_id, start_date=start_date)
    features_df = add_technical_indicators(merged_df.copy(), price_col='Close')

    # 2. Define and Shift Target (Future Close Price)
    TARGET_COL = 'Close_Future'
    features_df[TARGET_COL] = features_df['Close'].shift(-predict_ahead)
    features_df.dropna(inplace=True) # Drop last rows with NaN target

    # 3. Define Features (Using all engineered and raw data except target)
    # Note: We exclude 'Open', 'High', 'Low', 'Close', 'Volume' if we only want indicators, 
    # but for a strong baseline, we use everything.
    EXCLUDE_COLS = ['Close', TARGET_COL] # Exclude the current day's price and the target
    FEATURES = [col for col in features_df.columns if col not in EXCLUDE_COLS]
    
    X = features_df[FEATURES]
    y = features_df[TARGET_COL]
    
    # 4. Scaling Features (StandardScaler is often better for Tree-based models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler, FEATURES

# --- MODEL BUILDING AND TRAINING ---

def build_xgboost_model(params: dict = None):
    """Initializes and returns an XGBoost Regressor if available.

    Falls back to a `RandomForestRegressor` from scikit-learn if `xgboost`
    is not installed, allowing the pipeline to run in environments without
    XGBoost. Returns the model instance.
    """
    if _XGBOOST_AVAILABLE:
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        return xgb.XGBRegressor(**default_params)
    else:
        # Lazy import sklearn to avoid requiring XGBoost; this provides a reasonable fallback.
        from sklearn.ensemble import RandomForestRegressor
        default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'random_state': 42,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        return RandomForestRegressor(**default_params)

def run_xgboost_pipeline(ticker="MSFT", fred_series_id="DGS10", start_date="2015-01-01", 
                         test_ratio=0.2):
    """Runs the full XGBoost pipeline: Data -> Splitting -> Training -> Evaluation."""
    print(f"\n--- [1/4] Starting XGBoost Pipeline for {ticker} ---")
    
    # 1. Prepare Data
    X, y, scaler, features = prepare_tabular_data(ticker, fred_series_id, start_date)
    
    # 2. Train-Test Split (Standard split, not sequential like LSTM)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False, random_state=42 # shuffle=False maintains time-series order
    )

    print(f"\n--- [2/4] Building and Training Model ---")
    model = build_xgboost_model()
    
    # Train the model
    model.fit(X_train, y_train)

    print(f"\n--- [3/4] Evaluating Model ---")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"XGBoost Test RMSE (Original Price Scale): ${rmse:.4f}")

    # 4. Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top 5 Feature Importance ---")
    print(importance.head())

    # Save model (requires joblib or pickle for XGBoost)
    import joblib
    model_path = f"models/xgboost_model_{ticker}.joblib"
    joblib.dump(model, model_path)
    print(f"\n--- [4/4] Model Saved to {model_path} ---")

    return model, importance

if __name__ == "__main__":
    model, importance = run_xgboost_pipeline(ticker="MSFT")
    print("\nXGBoost Training Pipeline finished successfully.")