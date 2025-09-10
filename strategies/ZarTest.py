# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, IntParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, List, Optional, Union, Tuple
import talib.abstract as ta
from technical import qtpylib

class ZarTest(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = True  # Enable exit signals
    exit_profit_only = False  # Allow exits even if not profitable
    
    # ROI table:
    minimal_roi = {}
    # Stoploss:
    stoploss = -0.296
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True
    # Max Open Trades:
    max_open_trades = -1

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        dataframe[['bbl', 'bbm', 'bbu']] = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)[['lower', 'mid', 'upper']]
        
        # Add columns to track long and short signals
        dataframe['long_signal'] = (
            ((dataframe['dx'] > dataframe['mdi']) &
             (dataframe['adx'] > dataframe['mdi']) &
             (dataframe['pdi'] > dataframe['mdi'])) |
            (qtpylib.crossed_above(dataframe['close'], dataframe['bbu']))
        ).astype(int)
        
        dataframe['short_signal'] = (
            ((dataframe['dx'] > dataframe['mdi']) &
             (dataframe['adx'] > dataframe['pdi']) &
             (dataframe['mdi'] > dataframe['pdi'])) |
            (qtpylib.crossed_below(dataframe['close'], dataframe['bbl']))
        ).astype(int)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['mdi']) &
                (dataframe['pdi'] > dataframe['mdi'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')
        
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['bbu'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long Bollinger enter')
        
        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['pdi']) &
                (dataframe['mdi'] > dataframe['pdi'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')
        
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['close'], dataframe['bbl'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short Bollinger enter')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit short when long signal is detected
        dataframe.loc[
            (dataframe['long_signal'] == 1),
            'exit_short'] = 1
        
        # Exit long when short signal is detected
        dataframe.loc[
            (dataframe['short_signal'] == 1),
            'exit_long'] = 1
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                    current_rate: float, current_profit: float, **kwargs) -> float:
        # Additional custom exit logic
        # This method provides an extra layer of exit signal checking
        
        # If it's a short trade and a long signal is detected
        if trade.is_short and self.is_long_signal(pair):
            return 1  # Exit the short position
        
        # If it's a long trade and a short signal is detected
        if trade.is_long and self.is_short_signal(pair):
            return 1  # Exit the long position
        
        return 0  # No exit
    
    def is_long_signal(self, pair: str) -> bool:
        # Get the latest dataframe for the pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Check if the most recent row has a long signal
        if len(dataframe) > 0:
            return bool(dataframe.iloc[-1]['long_signal'])
        return False
    
    def is_short_signal(self, pair: str) -> bool:
        # Get the latest dataframe for the pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Check if the most recent row has a short signal
        if len(dataframe) > 0:
            return bool(dataframe.iloc[-1]['short_signal'])
        return False