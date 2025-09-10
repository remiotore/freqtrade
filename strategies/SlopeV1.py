import datetime
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy)
from scipy.stats import linregress

class SlopeV1(IStrategy):
    INTERFACE_VERSION = 3
    
    can_short = True
    timeframe = '30m'
    use_exit_signal = True
    exit_profit_only = True

    buy_params = {
        "window": 12,
    }

    sell_params = { }

    minimal_roi = {
        "0": 0.374,
        "180": 0.122,
        "274": 0.028,
        "847": 0
    }

    stoploss = -0.133

    trailing_stop = False

    max_open_trades = -1

    window = IntParameter(1, 30, space='buy', default=buy_params['window'])
    
    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
            },
            'subplots' : {
                 '%' : {
                    'open_pct' : { 'color' : 'blue' },
                    'close_pct' : { 'color' : 'green' },
                    'volume_pct' : { 'color' : 'red' },
                    'slope' : { 'color' : 'orange' },
                }
            }
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['open_pct']   = dataframe['open'].pct_change(self.window.value) * 100
        dataframe['close_pct']  = dataframe['close'].pct_change(self.window.value) * 100
        dataframe['volume_pct'] = dataframe['volume'].pct_change(self.window.value)
        dataframe['slope']      = dataframe['close'].rolling(window=self.window.value).apply(lambda x: linregress(range(len(x)), x).slope, raw=False)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (


                (dataframe['slope']          > 0) &
                (dataframe['slope'].shift(1) < 0) &
                (dataframe['volume']         > 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (


                (dataframe['slope']          < 0) &
                (dataframe['slope'].shift(1) > 0) &
                (dataframe['volume']         > 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (


                (dataframe['slope']          < 0) &
                (dataframe['slope'].shift(1) > 0) &
                (dataframe['volume']         > 0)
            ),
        'exit_long'] = 1

        dataframe.loc[
            (


                (dataframe['slope']          > 0) &
                (dataframe['slope'].shift(1) < 0) &
                (dataframe['volume']         > 0)
            ),
        'exit_short'] = 1

        return dataframe