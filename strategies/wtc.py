














import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy import DecimalParameter
from freqtrade.strategy import IStrategy
from pandas import DataFrame


import numpy as np  # noqa
import pandas as pd  # noqa
from sklearn import preprocessing




class wtc(IStrategy):















    buy_params = {
        "buy_max": 0.9609,
        "buy_max0": 0.8633,
        "buy_max1": 0.9133,
        "buy_min": 0.0019,
        "buy_min0": 0.0102,
        "buy_min1": 0.6864,
    }

    sell_params = {
        "sell_max": -0.7979,
        "sell_max0": 0.82,
        "sell_max1": 0.9821,
        "sell_min": -0.5377,
        "sell_min0": 0.0628,
        "sell_min1": 0.4461,
    }
    minimal_roi = {
        "0": 0.30873,
        "569": 0.16689,
        "3211": 0.06473,
        "7617": 0
    }
    stoploss = -0.128

    timeframe = '30m'

    buy_max = DecimalParameter(-1, 1, decimals=4, default=0.4393, space='buy')
    buy_min = DecimalParameter(-1, 1, decimals=4, default=-0.4676, space='buy')
    sell_max = DecimalParameter(-1, 1, decimals=4,
                                default=-0.9512, space='sell')
    sell_min = DecimalParameter(-1, 1, decimals=4,
                                default=0.6519, space='sell')

    buy_max0 = DecimalParameter(0, 1, decimals=4, default=0.4393, space='buy')
    buy_min0 = DecimalParameter(0, 1, decimals=4, default=-0.4676, space='buy')
    sell_max0 = DecimalParameter(
        0, 1, decimals=4, default=-0.9512, space='sell')
    sell_min0 = DecimalParameter(
        0, 1, decimals=4, default=0.6519, space='sell')

    buy_max1 = DecimalParameter(0, 1, decimals=4, default=0.4393, space='buy')
    buy_min1 = DecimalParameter(0, 1, decimals=4, default=-0.4676, space='buy')
    sell_max1 = DecimalParameter(
        0, 1, decimals=4, default=-0.9512, space='sell')
    sell_min1 = DecimalParameter(
        0, 1, decimals=4, default=0.6519, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        try:
            ap = (dataframe['high']+dataframe['low'] + dataframe['close'])/3

            esa = ta.EMA(ap, 10)

            d = ta.EMA((ap - esa).abs(), 10)
            ci = (ap - esa).div(0.0015 * d)
            tci = ta.EMA(ci, 21)

            wt1 = tci
            wt2 = ta.SMA(np.nan_to_num(wt1), 4)

            dataframe['wt1'], dataframe['wt2'] = wt1, wt2

            stoch = ta.STOCH(dataframe, 14)
            slowk = stoch['slowk']
            dataframe['slowk'] = slowk

            x = dataframe.iloc[:, 6:].values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            dataframe.iloc[:, 6:] = pd.DataFrame(x_scaled)


            dataframe['def'] = dataframe['slowk']-dataframe['wt1']

        except:
            dataframe['wt1'], dataframe['wt2'], dataframe['def'], dataframe['slowk'] = 0, 10, 100, 1000
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['wt1'], dataframe['wt2']))
                & (dataframe['wt1'].between(self.buy_min0.value, self.buy_max0.value))
                & (dataframe['slowk'].between(self.buy_min1.value, self.buy_max1.value))
                & (dataframe['def'].between(self.buy_min.value, self.buy_max.value))

            ),

            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['wt1'], dataframe['wt2']))
                & (dataframe['wt1'].between(self.sell_min0.value, self.sell_max0.value))
                & (dataframe['slowk'].between(self.sell_min1.value, self.sell_max1.value))
                & (dataframe['def'].between(self.sell_min.value, self.sell_max.value))

            ),
            'sell'] = 1
        return dataframe
