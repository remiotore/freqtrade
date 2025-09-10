
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import xgboost
import catboost
import sklearn
import pickle
from numba import jit
from scipy import signal

from freqtrade.strategy import IStrategy
import technical.indicators as ftt






import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class PS(IStrategy):

    INTERFACE_VERSION = 2

    minimal_roi = {
        "360": 0.0,
        "240": 0.05,
        "0": 0.1
    }


    stoploss = -0.9

    trailing_stop = False




    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 240

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def informative_pairs(self):

        return []

    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        verbose = False
        col_use = [
                    'volume','smadiff_3','smadiff_5','smadiff_8','smadiff_13',
                    'smadiff_21','smadiff_34','smadiff_55','smadiff_89',
                    'smadiff_120','smadiff_240','maxdiff_3','maxdiff_5','maxdiff_8',
                    'maxdiff_13','maxdiff_21','maxdiff_34','maxdiff_55','maxdiff_89',
                    'maxdiff_120','maxdiff_240','std_3','std_5','std_8','std_13',
                    'std_21','std_34','std_55','std_89','std_120','std_240',
                    'ma_3','ma_5','ma_8','ma_13','ma_21','ma_34','ma_55','ma_89',
                    'ma_120','ma_240','z_score_120','time_hourmin','time_dayofweek','time_hour' ]


        with open('user_data/notebooks/model_portfolio.pkl', 'rb') as f:
            model = pickle.load(f)
        model = model[0]


        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"smadiff_{i}"] = (dataframe['close'].rolling(i).mean() - dataframe['close'])

        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"maxdiff_{i}"] = (dataframe['close'].rolling(i).max() - dataframe['close'])

        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"maxdiff_{i}"] = (dataframe['close'].rolling(i).min() - dataframe['close'])

        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"std_{i}"] = dataframe['close'].rolling(i).std()

        for i in [3,5,8,13,21,34,55,89,120,240]:
            dataframe[f"ma_{i}"] = dataframe['close'].pct_change(i).rolling(i).mean()
        
        dataframe['z_score_120'] = ((dataframe.ma_13 - dataframe.ma_13.rolling(21).mean() + 1e-9) 
                            / (dataframe.ma_13.rolling(21).std() + 1e-9))
        
        dataframe["date"] = pd.to_datetime(dataframe["date"], unit='ms')
        dataframe['time_hourmin'] = dataframe.date.dt.hour * 60 + dataframe.date.dt.minute
        dataframe['time_dayofweek'] = dataframe.date.dt.dayofweek
        dataframe['time_hour'] = dataframe.date.dt.hour

        preds = pd.DataFrame(model.predict_proba(dataframe[col_use]))
        preds.columns = [f"pred{i}" for i in range(5)]
        dataframe = dataframe.reset_index(drop=True)
        dataframe = pd.concat([dataframe, preds], axis=1)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['pred4'] > .45) & 

                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['pred4'] < .28) 

                & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
