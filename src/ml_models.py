import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def prepare_ml_dataset(asset, returns, rolling_vol, ewma_vol):
    realized_vol = returns[asset].rolling(21).std().shift(-1) * np.sqrt(252)
    
    df_ml = pd.DataFrame()
    df_ml["return"] = returns[asset]
    df_ml["return_lag1"] = returns[asset].shift(1)
    df_ml["return_sq_lag1"] = returns[asset].shift(1)**2
    df_ml["rolling_vol"] = rolling_vol[asset]
    df_ml["ewma_vol"] = ewma_vol[asset]
    df_ml["target"] = realized_vol
    
    return df_ml.dropna()


def train_models(df_ml):
    split = int(len(df_ml) * 0.8)
    
    train = df_ml.iloc[:split]
    test = df_ml.iloc[split:]
    
    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    return rmse_lr, rmse_rf, y_test.index
