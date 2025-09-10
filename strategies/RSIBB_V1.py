import talib.abstract as ta
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy


class RSIBB_V1(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    use_exit_signal = False
    
    # ROI table:
    minimal_roi = {
        "0": 0.158,
        "32": 0.046,
        "81": 0.01,
        "164": 0
    }

    # Stoploss:
    stoploss = -0.262

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.066
    trailing_stop_positive_offset = 0.087
    trailing_only_offset_is_reached = True

    @property
    def plot_config(self):
        return {
            'main_plot': {
                'bbu' : { 'color' : 'blue' },
                'bbm' : { 'color' : 'orange' },
                'bbl' : { 'color' : 'blue' },
            },
            'subplots': {
                "RSI": {
                    'rsi_fast': {'color': 'yellow'},
                    'rsi_slow': {'color': 'red'},
                },
            }
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=6)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=12)
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bbl'] = bollinger['lower']
        dataframe['bbm'] = bollinger['mid']
        dataframe['bbu'] = bollinger['upper']
        dataframe['bb_width'] = dataframe['bbu'] - dataframe['bbl']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi_fast"], 70)) &
                (qtpylib.crossed_above(dataframe["close"], dataframe['bbu'])) &
                (dataframe["bb_width"].diff() > 0) &
                (dataframe["rsi_fast"].diff() > 0) &
                (dataframe["rsi_slow"].diff() > 0) 
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe["rsi_fast"], 70))
            ),
            "exit_long"
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 3