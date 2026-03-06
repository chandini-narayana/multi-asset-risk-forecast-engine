# Multi-Asset Risk Forecasting Engine

A hybrid **quantitative finance + machine learning system** that forecasts asset volatility and portfolio risk using statistical models and ML-based volatility prediction.

---

## Overview

This project builds a multi-asset risk engine that models:

• asset volatility  
• cross-asset covariance  
• portfolio risk dynamics  
• forward-looking volatility forecasts

Traditional statistical risk models are combined with machine learning to improve volatility predictions.

---

## Features

Statistical Risk Engine

- Log return modeling
- Rolling volatility estimation
- EWMA volatility modeling
- Covariance matrix estimation
- Portfolio volatility calculation
- Forecast vs realized volatility backtesting

Machine Learning Layer

- Linear Regression volatility forecasting
- Random Forest volatility forecasting
- Feature engineering using lagged returns and volatility
- Multi-asset model evaluation using RMSE

Final System Output

- Asset-level volatility forecasts
- Portfolio-level risk estimation

---

## Project Structure
multi-asset-risk-forecast-engine
│
├── notebooks
│ ├── 01_data_exploration.ipynb
│ ├── 02_risk_engine.ipynb
│ └── 03_ml_volatility.ipynb
│
├── src
│ ├── risk_metrics.py
│ └── ml_models.py
│
├── requirements.txt
└── README.md

---

## Installation

Clone the repository: git clone https://github.com/chandini-narayana/multi-asset-risk-forecast-engine.git

Install dependencies: pip install -r requirements.txt


Run notebooks using Jupyter.

---

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-Learn  
Matplotlib  
Seaborn  
yfinance

---

## Hardware Requirements

CPU based system  
8GB RAM recommended  
No GPU required

---

## Example Assets Used

- Apple (AAPL)
- Microsoft (MSFT)
- SPY ETF
- Bitcoin (BTC-USD)
- Gold ETF (GLD)

---

## Future Improvements

- GARCH volatility modeling
- Portfolio optimization integration
- Risk regime detection
- Dashboard visualization

---

## Author

Chandini Narayana

## Project Structure
