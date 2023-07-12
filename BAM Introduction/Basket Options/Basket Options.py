import pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf
import numpy as np
import math
from tqdm import tqdm


def backtest_basket(strikes, tickers_basket):
    # Here we will consider a business month expiration strategy (21 day in a business day convention).
    df_backtest = yf.download(tickers=tickers_basket)["Close"].pct_change(periods=21).apply(lambda x: x + 1)
    df_backtest.dropna(inplace=True)
    counter = 0
    # We are going to check if every condition
    while counter < len(strikes):
        # Initialization
        if counter == 0:
            # Down Condition
            if strikes[counter] <= 100:
                mask = df_backtest.iloc[:, counter] <= strikes[counter] / 100
            else:
                mask = df_backtest.iloc[:, counter] >= strikes[counter] / 100
        else:
            if strikes[counter] <= 100:
                mask = np.logical_and(mask, df_backtest.iloc[:, counter] <= strikes[counter] / 100)
            else:
                mask = np.logical_and(mask, df_backtest.iloc[:, counter] >= strikes[counter] / 100)
        counter += 1
    df_backtest["Profit"] = mask
    profits_proportion = sum(df_backtest["Profit"] / len(df_backtest))
    return df_backtest, profits_proportion


def simulate_paths(tickers, NTS, T, N=100):
    """
    Simulating realizations of log-normal risk-neutral random walk

    :param tickers: tickers we need to simulate asset path on a given period of time
    :param NTS: number of time steps
    :param T: Derivatives expiration (in years)
    :param N: Number of simulated paths for each stock

    :return:
    """

    # Step 1: We download targeted assets price (start at mid 2020 for COVID crisis).
    df = yf.download(tickers=tickers, start="2020-06-01")["Close"]
    # Step 2: Compute annualized historical drift and covariance matrix.
    drift = df.pct_change().mean() * 252
    cov_matrix = df.pct_change().cov() * math.sqrt(252)
    # Step 3: Initializing the dictionary containing for each stock the realized path.
    realization_dict = {ticker: np.zeros((NTS, N)) for ticker in tickers}
    dt = T / NTS
    # We will start at current price for every simulation.
    for ticker in tickers:
        S0 = df[ticker][-1]
        realization_dict[ticker][0, :] = [S0 for simulation in range(N)]
    # Everything has been initialized, let us perform the simulations.
    for simulation in tqdm(range(N)):
        for timestep in range(1, NTS):
            # For every path we need a multivariate normal random variable with same covariance matrix as stocks.
            random_variable = np.random.multivariate_normal(np.zeros(len(tickers)), cov_matrix)
            # Now every stock will follow the Black and Scholes model.
            for ticker in range(len(tickers)):
                realization_dict[tickers[ticker]][timestep, simulation] = realization_dict[tickers[ticker]][
                                                                              timestep - 1, simulation] \
                                                                          * math.exp(
                    (drift[ticker] - 1 / 2 * cov_matrix.iloc[ticker, ticker] ** 2) * dt
                    + math.sqrt(dt) * random_variable[
                        ticker])
    return realization_dict


if __name__ == "__main__":
    number_paths = 80
    tickers = ["AFL", "ZION"]
    simulate = simulate_paths(tickers=tickers, NTS=1000, T=1 / 12, N=number_paths)
    strikes = [99, 102]
    # We will check for every path if the condition is met.
    counter = 0
    for path in range(number_paths):
        print("Veryfing path number", path)
        for i in range(len(tickers)):
            ticker = tickers[i]
            strike = strikes[i]
            if strikes[i] <= 100:
                # On every path we will check the returns made by the asset on one month.
                if (simulate[ticker][-1, path] - simulate[ticker][0, path]) / simulate[ticker][0, path] >= (
                        strike - 100) / 100:
                    print("DOWN", (simulate[ticker][-1, path] - simulate[ticker][0, path]) / simulate[ticker][0, path])
                    break
            else:
                if (simulate[ticker][-1, path] - simulate[ticker][0, path]) / simulate[ticker][0, path] <= (
                        strike - 100) / 100:
                    print("UP", (simulate[ticker][-1, path] - simulate[ticker][0, path]) / simulate[ticker][0, path])
                    break
            if i ==(len(tickers) - 1):
                print("This path works well! ", path)
                counter += 1



