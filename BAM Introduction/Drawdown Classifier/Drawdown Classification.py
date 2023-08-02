import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from tqdm import tqdm
import math
import requests


def identifying_drawdowns(ticker="^GSPC", drawdown_threshold=0.15, local_max_period=60, plotting=True, mocked = True):
    """
    Draw-down as periods in which prices are lower by more than drawdown_threshold with respect to a local max
    (by default rolling period of 60D)
    :param mocked: Boolean. Check if stock sata cna be used locally
    :param ticker:
    :param drawdown_threshold:
    :param local_max_period:
    :param plotting:
    :return:
    """
    if not mocked:
        # We download price data.
        df = yf.download(ticker)
        df.to_csv("SP500.csv")
    else:
        df = pd.read_csv("SP500.csv", index_col = 0)

    # Finding local max on a given period, 1M per default (Looking for max in last 30D period)
    df['Rolling_Max'] = df['Close'].rolling(window=local_max_period, min_periods=1).max()

    # We will initialize a column as a reference point marking the first moment of a draw-down as a local max.
    df['Reference_Point'] = False

    # Local max are going to be the reference point for every draw-down. (Days with a local max)
    df.loc[df['Close'] == df['Rolling_Max'], 'Reference_Point'] = True

    # Computing draw-down in comparison with local max.
    df['Drawdown'] = (df['Close'] / df['Rolling_Max']) - 1
    df.loc[df['Reference_Point'], 'Drawdown'] = None

    # Identify draw-down.
    df['Is_Drawdown'] = df['Drawdown'] < -drawdown_threshold

    # Once a draw-down is identified, the starting point will be the local max, not the moment the threshold is
    # trespassed
    for i in tqdm(range(1, len(df))):
        if df.iloc[i]["Reference_Point"]:
            starting_point = df.index.tolist()[i]
        if np.logical_and(df.iloc[i]["Is_Drawdown"], not df.iloc[i - 1]["Is_Drawdown"]):
            df.loc[starting_point:df.index.tolist()[i], 'Is_Drawdown'] = True

    # Identify draw-down period.
    df['Drawdown_Period'] = df['Is_Drawdown'].diff().cumsum()
    df.drop(columns=["Open", "High", "Low", "Adj Close", "Volume"], inplace=True)
    df['Date'] = df.index.tolist()
    # Calcula la duración y magnitud de cada drawdown
    drawdowns = df[df['Is_Drawdown']].groupby('Drawdown_Period').agg(
        Start_Date=('Date', 'first'),
        End_Date=('Date', 'last'),
        Duration=('Date', 'count'),
        Magnitude=('Drawdown', 'min')
    )
    """"
    real_dates = []
    real_period = []
    for starting_date, ending_date in zip(drawdowns["Start_Date"].tolist(), drawdowns["End_Date"].tolist()):
        max_reference = df.loc[starting_date, "Rolling_Max"]
        mask = df["Close"] == max_reference
        real_dates.append(df[mask]["Date"].values[0])
        real_period.append(len(df.loc[df[mask]["Date"].values[0]:ending_date]))

    drawdowns["Start_Date"] = real_dates
    drawdowns["Duration"] = real_period
    """
    vols = []
    for start, end in list(zip(drawdowns["Start_Date"].to_list(), drawdowns["End_Date"].to_list())):
        vols.append(math.sqrt(252)*df.loc[start:end, "Close"].pct_change().std())
    drawdowns["Volatility"] = vols

    if plotting:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode='lines', name='Prices'))

        for starting_date, ending_date in zip(drawdowns["Start_Date"].tolist(), drawdowns["End_Date"].tolist()):
            fig.add_shape(
                type='rect',
                xref='x',
                yref='y',
                x0=starting_date,
                y0=min(df["Close"].values),
                x1=ending_date,
                y1=max(df["Close"].values),
                fillcolor='gray',
                opacity=0.2,
                layer='below',
            )

        # Personaliza el gráfico
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price',
            title=f'{ticker} Prices and draw-down',
        )

        # Muestra el gráfico
        fig.show()
        return drawdowns, df


if __name__ == "__main__":
    drawdowns, df = identifying_drawdowns(local_max_period=60, mocked = True)
