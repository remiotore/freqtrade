# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

class SMA(IStrategy):
    stoploss = -0.05

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        data = dataframe.copy()
        data['returns'] = data['close'] - data['close'].shift()
        data.dropna(inplace=True)
        data['volumeGap'] = data['volume'] / data['volume'].shift()
        data.dropna(inplace=True)
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
        endog = data['returns']
        exog = data [['volumeGap']]
        # Fit the 3-regime model
        mod_2 = sm.tsa.MarkovRegression(endog=endog, exog=exog, k_regimes=2, order=2)
        res_2 = mod_2.fit(search_reps=20)
        
        ## uncomment to plot prob's
##        fig, axes = plt.subplots(3, figsize=(10,7))
##        ax = axes[0]
##        ax.plot(res_2.smoothed_marginal_probabilities[0])
##        ax.set(title='Smoothed probability of down regime')
##        ax = axes[1]
##        ax.plot(res_2.smoothed_marginal_probabilities[1])
##        ax.set(title='Smoothed probability of up regime')
##        ax = axes[2]
##        ax.plot(data.close)
##        ax.set(title='Returns')
##        plt.tight_layout()
##        plt.show()
        dataframe["prob"] = res_2.smoothed_marginal_probabilities[1]        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['prob'] < 0.45)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['prob'] > 0.45)

            ),
            'sell'] = 1
        return dataframe

