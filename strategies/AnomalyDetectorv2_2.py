import numpy as np
from pandas import DataFrame
import pandas as pd
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import talib.abstract as ta

class AnomalyDetectorv2_2(IStrategy):
    def version(self) -> str:
        return "AnomalyDetector-v2-1h"

    INTERFACE_VERSION = 3

    """
    Strategy to detect volume anomalies using Z-Score
    """   

    stoploss = -0.99
    timeframe = '1h'

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = False
    
    fast_length = IntParameter(10, 20, default=12, space='buy')
    slow_length = IntParameter(24, 30, default=26, space='buy')
    signal_length = IntParameter(7, 14, default=9, space='buy')
    rsi_length = IntParameter(10, 30, default=14, space='buy')
    atr_length = IntParameter(10, 30, default=14, space='buy')
    
    plot_config = {

        'main_plot': {
        },
        'subplots': {

            "TVTechRec": {
                'tvsummaryaction': {'color': 'white'}
            },
            "ATR": {
                'atr': {'color': 'white'}
            },
            "RSI": {
                'rsi': {'color': 'purple'},
                'rsi_backup': {'color': 'red'},
            },
            "Volume": {
                'volume': {'color': 'black'},
                'anomaly': {
                    'color': 'orange',
                    'drawstyle': 'steps-mid',
                    'fill_to': 'anomaly',
                    'panel': 'lower',
                    'linestyle': '-',
                    'linewidth': 2,
                },
            },
            "Z-Score": {
                'vol_z_score': {'color': 'blue'},
            },
        }
    }

    startup_candle_count = 999

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_length.value)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_length.value)

        vol_z_score = (dataframe['volume'] - dataframe['volume'].rolling(window=30).mean()) / dataframe['volume'].rolling(window=30).std()
        threshold = 3
        dataframe['anomaly'] = np.where(vol_z_score > threshold, 1, 0)

        delta = dataframe['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_length.value).mean()
        avg_loss = loss.rolling(window=self.rsi_length.value).mean()

        rs = avg_gain / avg_loss

        dataframe['rsi_backup'] = 100 - (100 / (1 + rs)) #For exchanges where RSI shows zero using ta-lib

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['anomaly'] > 0) &
                (dataframe['atr'] > dataframe['atr'].rolling(window=self.atr_length.value).mean())
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe['rsi'] > 70) | (dataframe['rsi_backup'] > 85))
            ),
            'exit_long'] = 1
        return dataframe