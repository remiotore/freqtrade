# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time

class Markov(IStrategy):
    stoploss = -0.05

    # Optimal timeframe for the strategy
    timeframe = '30m'

    # run 'populate_indicators' only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        data = dataframe.copy()
        frame_size = 96  # 2 days
        len_df = len(dataframe)
        data['returns'] = np.log(data['close'] / data['close'].shift())

        data['volumeGap'] = data['volume'] / data['volume'].shift()
        data['fractHigh'] = (data['high'] - data['open']) / data['open']
        data['fractLow'] = (data['open'] - data['low']) / data['open']
        data.dropna(inplace=True)
        data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

        endog = data['returns'].values
        exog = data[['volumeGap', 'fractHigh', 'fractLow']].values

        start_time = time.time()
        for i in range(len_df):
            if i + frame_size < len_df:
                print_frequency = 10
                if i and i % print_frequency == 0:
                    print('----------------------')
                    print(f'{i} steps of {len_df}')
                    curr_time = time.time()
                    time_passed = round((curr_time - start_time) / 60, 2)
                    approx_mins = round((time_passed / (i / print_frequency)) * len_df / print_frequency, 2)
                    print(f'Total time passed: {time_passed} min.')
                    print(f'Approx time to finish: {approx_mins} min.')
                    print('----------------------')
                # Fit the 3-regime model
                mod_2 = sm.tsa.MarkovRegression(endog=endog[i: i + frame_size], exog=exog[i: i + frame_size], k_regimes=2)
                res_2 = mod_2.fit(search_reps=50)
                # print(len(res_2.smoothed_marginal_probabilities))
                dataframe.at[i + frame_size, 'prob_down'] = res_2.smoothed_marginal_probabilities[0][-1]
                dataframe.at[i + frame_size, 'prob_side'] = res_2.smoothed_marginal_probabilities[1][-1]
                dataframe.at[i + frame_size, 'prob_up'] = res_2.smoothed_marginal_probabilities[2][-1]

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

