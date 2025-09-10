# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import numpy as np

"""
=============== SUMMARY METRICS ================
| Metric                | Value                |
|-----------------------+----------------------|
| Backtesting from      | 2019-01-01 00:00:00  |
| Backtesting to        | 2021-02-27 19:00:00  |
| Max open trades       | 10                   |
|                       |                      |
| Total trades          | 9967                 |
| Total Profit %        | 684.72%              |
| Trades per day        | 12.65                |
|                       |                      |
| Best Pair             | ADADOWN/USDT 508.79% |
| Worst Pair            | SNX/USDT -88.76%     |
| Best trade            | HBAR/USDT 74.59%     |
| Worst trade           | TRXDOWN/USDT -6.05%  |
| Best day              | 567.76%              |
| Worst day             | -133.79%             |
| Days win/draw/lose    | 410 / 16 / 362       |
| Avg. Duration Winners | 11:52:00             |
| Avg. Duration Loser   | 15:08:00             |
|                       |                      |
| Abs Profit Min        | -62.306 USDT         |
| Abs Profit Max        | 6913.458 USDT        |
| Max Drawdown          | 471.86%              |
| Drawdown Start        | 2019-06-30 01:00:00  |
| Drawdown End          | 2019-09-01 19:00:00  |
| Market change         | 431.28%              |
================================================
"""
def supertrend(dataframe, multiplier=3, period=10):
    """
    Supertrend Indicator
    adapted for freqtrade
    from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """
    df = dataframe.copy()

    df['TR'] = ta.TRANGE(df)
    df['ATR'] = df['TR'].ewm(alpha=1 / period).mean()

    # atr = 'ATR_' + str(period)
    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)

    # Compute basic upper and lower bands
    df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] > df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] < df['final_lb'].iat[i] else 0.00

    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    # df.to_csv('user_data/Supertrend.csv')
    return DataFrame(index=df.index, data={
        'ST': df[st],
        'STX': df[stx]
    })


class SuperTrend(IStrategy):

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"

    max_open_trades: int = 20

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    minimal_roi = {
        "0": 100
    }

    stoploss = -0.05  # -10.0
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_only_offset_is_reached = True
    trailing_stop_positive_offset = 0.05  # 0.1

    # Optimal timeframe for the strategy
    # timeframe = '5m'  # 15m - alternative
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["supertrend_3_12"] = supertrend(dataframe, 3, 12)["STX"]
        dataframe["supertrend_1_10"] = supertrend(dataframe, 1, 10)["STX"]
        dataframe["supertrend_2_11"] = supertrend(dataframe, 2, 11)["STX"]

        # required for graphing
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe["supertrend_3_12"] == "up") &
                        (dataframe["supertrend_1_10"] == "up") &
                        (dataframe["supertrend_2_11"] == "up")
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe["supertrend_3_12"] == "down") &
                        (dataframe["supertrend_1_10"] == "down") &
                        (dataframe["supertrend_2_11"] == "down")
                 )
            ),
            'sell'] = 1
        return dataframe