import talib.abstract as ta
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter, informative, merge_informative_pair
from datetime import datetime
from freqtrade.persistence import Trade



class EMARibbonStrategy(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '5m'

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.07 # set stoploss to 7% as default

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = False

    use_custom_stoploss = True


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, dataframe: DataFrame, **kwargs) -> float:

        if qtpylib.crossed_above(dataframe['ema10'], dataframe['ema25']):
            dataframe['bullish_market'] = True

        if qtpylib.crossed_below(dataframe['ema10'], dataframe['ema25']):
            dataframe['bearish_market'] = True

        if dataframe['bullish_market'] == True:
            stoploss = 0.10  # set stoploss to 10% for a bullish market
        elif dataframe['bearish_market'] == True:
            stoploss = 0.05  # set stoploss to 5% for a bearish market

        return stoploss


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for length in [10, 15, 20, 25]:
            dataframe[f'ema{length}'] = ta.EMA(dataframe, length)

        dataframe['rsi'] = ta.RSI(dataframe)

        return dataframe


    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        conditions = []

        conditions.append(
            (dataframe['volume'] > 0) & # Buy when volume > 0
            (qtpylib.crossed_below(dataframe['ema10'], dataframe['ema20'])) # Buy when ema10 crosses below ema20
        )

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        conditions = []

        conditions.append(
            (dataframe['volume'] > 0) & # Sell when volume > 0
            (qtpylib.crossed_above(dataframe['ema10'], dataframe['ema25'])) # Sell when ema10 crosses above ema25
        )

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
