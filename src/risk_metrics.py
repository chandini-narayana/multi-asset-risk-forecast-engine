import numpy as np
import pandas as pd


def compute_log_returns(prices):
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()


def rolling_volatility(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252)


def ewma_volatility(returns, lambda_=0.94):
    ewma_vol = pd.DataFrame(index=returns.index, columns=returns.columns)

    for col in returns.columns:
        var = returns[col].iloc[0] ** 2
        ewma_series = []

        for r in returns[col]:
            var = lambda_ * var + (1 - lambda_) * r ** 2
            ewma_series.append(var)

        ewma_vol[col] = np.sqrt(ewma_series) * np.sqrt(252)

    return ewma_vol


def ewma_covariance(returns, lambda_=0.94):
    cov_matrix = returns.cov().values
    cov_matrices = []

    for i in range(len(returns)):
        r_t = returns.iloc[i].values.reshape(-1, 1)
        cov_matrix = lambda_ * cov_matrix + (1 - lambda_) * (r_t @ r_t.T)
        cov_matrices.append(cov_matrix)

    return cov_matrices


def portfolio_volatility(cov_matrix, weights):
    var = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(var) * np.sqrt(252)