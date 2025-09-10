from pandas import DataFrame
import talib.abstract as ta
from functools import reduce
import numpy as np
from freqtrade.strategy import (IStrategy)

# DIV v2.0 - 2022-04-22
# by Sanka 

class DIV_v2(IStrategy):
    minimal_roi = {
        "0": 0.05
    }

    stoploss = -0.05

    timeframe = '5m'
    startup_candle_count = 200
    process_only_new_candles = True

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    plot_config = {
        "main_plot": {
            "ohlc_bottom" : {
                "type": "scatter",
                'plotly': {
                    "mode": "markers",
                    "name": "a",
                    "text": "aa",
                    "marker": {
                        "symbol": "cross-dot",
                        "size": 3,
                        "color": "black"
                    }
                }
            },
        },
        "subplots": {
            "rsi": {
                "rsi": {"color": "blue"},
                "rsi_bottom" : {
                    "type": "scatter",
                    'plotly': {
                        "mode": "markers",
                        "name": "b",
                        "text": "bb",
                        "marker": {
                            "symbol": "cross-dot",
                            "size": 3,
                            "color": "black"
                        }
                    }
                },
            }
        }
    }

    #############################################################

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Divergence
        dataframe = divergence(dataframe, "rsi", 14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bullish_divergence_rsi"] == True) &
                (dataframe['rsi'] < 30) &
                (dataframe["volume"] > 0)
            ), 'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


def divergence(dataframe: DataFrame, source='rsi', length=14):
    # Detect divergence between close price and source

    # Detect HL or LL
    high_low = dataframe.copy()
    high_low['ohlc_bottom'] = np.NaN
    high_low.loc[(dataframe['close'].shift() <= dataframe['close'].shift(2)) & (dataframe['close'] >= dataframe['close'].shift()), 'ohlc_bottom'] = dataframe['close'].shift()
    high_low["ohlc_bottom"].fillna(method='ffill', inplace=True)
    high_low['rsi_bottom'] = np.NaN
    high_low.loc[(dataframe[source].shift() <= dataframe[source].shift(2)) & (dataframe[source] >= dataframe[source].shift()), 'rsi_bottom'] = dataframe[source].shift()
    high_low["rsi_bottom"].fillna(method='ffill', inplace=True)

    high_low['ohlc_top'] = np.NaN
    high_low.loc[(dataframe['close'].shift() >= dataframe['close'].shift(2)) & (dataframe['close'] <= dataframe['close'].shift()), 'ohlc_top'] = dataframe['close'].shift()
    high_low["ohlc_top"].fillna(method='ffill', inplace=True)
    high_low['rsi_top'] = np.NaN
    high_low.loc[(dataframe[source].shift() >= dataframe[source].shift(2)) & (dataframe[source] <= dataframe[source].shift()), 'rsi_top'] = dataframe[source].shift()
    high_low["rsi_top"].fillna(method='ffill', inplace=True)

    # Detect divergence
    dataframe[f'bullish_divergence_{source}'] = False
    for i in range(5, (length+1)):
        # Check there is nothing between the 2 diverging points
        conditional_array = []
        for ii in range(1, i):
            conditional_array.append(high_low["ohlc_bottom"].shift(i).le(high_low['ohlc_bottom'].shift(ii)))
        res = reduce(lambda x, y: x & y, conditional_array)
        dataframe.loc[(
            (high_low["ohlc_bottom"].lt(high_low['ohlc_bottom'].shift(i))) &
            (high_low["rsi_bottom"].gt(high_low['rsi_bottom'].shift(i))) &
            (high_low["ohlc_bottom"].le(high_low['ohlc_bottom'].shift())) &
            (res)
            ), f"bullish_divergence_{source}"] = True

    dataframe[f'hidden_bearish_divergence_{source}'] = np.NaN
    for i in range(5, (length+1)):
        # Check there is nothing between the 2 diverging points
        conditional_array = []
        for ii in range(1, i):
            conditional_array.append(high_low["ohlc_top"].shift(i).ge(high_low['ohlc_top'].shift(ii)))
        res = reduce(lambda x, y: x & y, conditional_array)
        dataframe.loc[(
            (high_low["ohlc_top"].gt(high_low['ohlc_top'].shift(i))) &
            (high_low["rsi_top"].lt(high_low['rsi_top'].shift(i))) &
            (high_low["ohlc_top"].ge(high_low['ohlc_top'].shift())) &
            (res)
            ), f"hidden_bearish_divergence_{source}"] = True

    return dataframe