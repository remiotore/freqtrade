
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import time

class MarkovV4(IStrategy):
    stoploss = -0.05

    timeframe = '30m'

    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    def bot_loop_start(self, **kwargs):
        pass
        headers = ["date", "open", "high", "low", "close", "volume"]
        filename = os.path.join("user_data/data/binance/BTC_USDT-30m.json")
        data = pd.read_json(filename)
        data.columns = headers

        data['returns'] = np.log(data['close'] / data['close'].shift())

        model = sm.tsa.MarkovRegression(data['returns'][-2500:], k_regimes=3, switching_variance=True)
        res_1 = model.fit(search_reps=50)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
	
        dataframe['volumeGap'] = np.log(dataframe['volume'] / dataframe['volume'].shift())
        dataframe['dailyChange'] = (dataframe['close'] - dataframe['open']) / dataframe['open']
        dataframe['fractHigh'] = (dataframe['high'] - dataframe['open']) / dataframe['open']
        dataframe['fractLow'] = (dataframe['open'] - dataframe['low']) / dataframe['open']
        dataframe['forecastVariable'] = dataframe['close'].shift(-1) - dataframe['close']

        dataframe = dataframe.fillna(0)


        endog = dataframe['forecastVariable']
        exog = dataframe[['volumeGap', 'dailyChange', 'fractHigh', 'fractLow']]

        mod_2 = sm.tsa.MarkovRegression(endog=endog, k_regimes=3, exog=exog)
        res_2 = mod_2.fit(search_reps=50)

        dataframe['prob_down'] = res_2.smoothed_marginal_probabilities[0]
        dataframe['prob_side'] = res_2.smoothed_marginal_probabilities[1]
        dataframe['prob_up'] = res_2.smoothed_marginal_probabilities[2]

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['prob_up'] > 0.8)
                &
                (dataframe['prob_side'] < 0.5)
                &
                (dataframe['prob_down'] < 0.5)

            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['prob_up'] < 0.5)
                &
                (dataframe['prob_side'] < 0.5)
                &
                (dataframe['prob_down'] > 0.8)

            ),
            'sell'] = 1
        return dataframe

