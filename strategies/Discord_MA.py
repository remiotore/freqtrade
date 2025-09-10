# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
# --------------------------------


class MA(IStrategy):

    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.15,    
        "20160": 0.15
    }
    
    #can_short = True

    # Optimal stoploss designed for the strategy
    stoploss = -0.025
    trailing_stop = True

    # Optimal timeframe for the strategy
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
             (qtpylib.crossed_above(dataframe['ema8'], dataframe['ema21']))
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            (qtpylib.crossed_below(dataframe['ema8'], dataframe['ema21']))
            ),
            'exit_long'] = 1
        return dataframe