# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.strategy import IStrategy
from datetime import datetime
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib



class ZaratustraV24(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = True
    
    # ROI table:
    minimal_roi = {
        "0": 0.100,
        "30": 0.090,
        "60": 0.040,
        "127": 0
    }

    # Stoploss:
    stoploss = -0.2

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.080
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 10

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {
            'EMA' : {} 
        }
        plot_config['subplots'] = {
            'DI': {
                'DX' : { 'color': 'yellow' },
                'ADX': { 'color': 'orange' },
                'PDI': { 'color': 'green' },
                'MDI': { 'color': 'red' },
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['EMA'] = ta.EMA(dataframe)
        dataframe['LRS'] = ta.LINEARREG_SLOPE(dataframe)

        dataframe['DX']  = ta.DX(dataframe)       # ta.SMA(      ta.DX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['ADX'] = ta.ADX(dataframe)      # ta.SMA(     ta.ADX(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['PDI'] = ta.PLUS_DI(dataframe)  # ta.SMA( ta.PLUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])
        dataframe['MDI'] = ta.MINUS_DI(dataframe) # ta.SMA(ta.MINUS_DI(dataframe) * dataframe['volume']) / ta.SMA(dataframe['volume'])

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['DX'], dataframe['PDI'])) &
                (dataframe['PDI'] > dataframe['MDI']) &
                (dataframe['LRS'] > dataframe['LRS'].shift(1))
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['DX'], dataframe['MDI'])) &
                (dataframe['MDI'] > dataframe['PDI']) &
                (dataframe['LRS'] < dataframe['LRS'].shift(1))
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['DX'], dataframe['ADX'])) |
                (qtpylib.crossed_below(dataframe['DX'], dataframe['PDI'])) |
                (dataframe['DX'] < dataframe['PDI'])
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'Long DI exit')

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['DX'], dataframe['ADX'])) |
                (qtpylib.crossed_below(dataframe['DX'], dataframe['MDI'])) |
                (dataframe['DX'] < dataframe['MDI'])
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'Short DI exit')
        
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10