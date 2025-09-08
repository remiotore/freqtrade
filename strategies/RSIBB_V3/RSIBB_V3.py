import talib.abstract as ta
import numpy as np
import pandas as pd
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy, informative
from freqtrade.persistence import Trade


class RSIBB_V3(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    
    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.2

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "bbu": {"color": "blue"},
                "bbm": {"color": "orange"},
                "bbl": {"color": "blue"},
            },
            "subplots": {
                "Volume" : {
                    "volume" : {"color": "red"},
                    "vbbu" : {"color": "blue"},
                    "vbbm" : {"color": "orange"},
                    "vbbl" : {"color": "blue"},
                },
                "RSI" : {
                    "rsi_fast" : {"color": "yellow"},
                    "rsi_slow" : {"color": "orange"},
                },
            },
        }
        
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=6)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=12)
        
        # Bollinger bands on price
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bbl"] = bollinger["lower"]
        dataframe["bbm"] = bollinger["mid"]
        dataframe["bbu"] = bollinger["upper"]
        
        # Bollinger bands on volume
        bollinger = qtpylib.bollinger_bands(dataframe["volume"], window=20, stds=2)
        dataframe["vbbl"] = bollinger["lower"]
        dataframe["vbbm"] = bollinger["mid"]
        dataframe["vbbu"] = bollinger["upper"]
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["close"], dataframe["bbu"]) &
                qtpylib.crossed_above(dataframe["volume"], dataframe["vbbu"])
            ),
            "enter_long"
        ] = 1
        
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe["close"], dataframe["bbl"]) &
                qtpylib.crossed_below(dataframe["volume"], dataframe["vbbl"])
            ),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 1