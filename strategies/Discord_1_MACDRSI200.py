# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta


class MACDRSI200(IStrategy):

    ticker_interval = '5m'

    # Buy hyperspace params:
    buy_params = {
     'buy-rsi-value': 43
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-rsi-value': 83
    }

    # ROI table:
    minimal_roi = {
        "0": 0.03551,
        "290": 0.03429,
        "573": 0.03082,
        "820": 0.02587,
        "935": 0.02257,
        "1232": 0.01818,
        "1476": 0.01692,
        "1731": 0.01224,
        "2001": 0.00764,
        "2150": 0.00435,
        "2307": 0
    }

    # Stoploss:
    stoploss = -0.04562

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['sell-rsi'] = ta.RSI(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'].rolling(8).min() < 43) &
                    (dataframe['close'] > dataframe['ema200']) &
                    (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']))
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'].rolling(8).max() > 83) &
                    (dataframe['macd'] > 0) &
                    (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))
            ),
            'sell'] = 1

        return dataframe
