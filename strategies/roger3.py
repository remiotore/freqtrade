import numpy as np
import pandas as pd
import pandas_ta as ta
from pandas_ta import ema
import talib.abstract as talib
from talib import ATR, RSI
from technical import qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime
import logging  # remove after
logger = logging.getLogger(__name__)  # remove after

class roger3(IStrategy):

    INTERFACE_VERSION = 2

    timeframe = '15m'

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.1

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['ema50'] = ema(dataframe['close'], length=50)
        dataframe['ema200'] = ema(dataframe['close'], length=200)

        dataframe['atr'] = ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)

        dataframe['rsi'] = RSI(dataframe['close'], timeperiod=14)
        
        return dataframe

   
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            
            dataframe.loc[
                (

                    (dataframe['ema200'] > dataframe['ema50'])

                    & (dataframe['ema20'] > dataframe['ema20'].shift(1))
                ),
                'buy'
            ] = 1
    
            return dataframe
    
    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define the sell conditions incorporating trend reversal, profit protection, 
        volatility adjustment, and a time-based component.
        """
        dataframe.loc[
            (

                (dataframe['ema50'] < dataframe['ema200']) &
                (dataframe['rsi'] > 70) | # Overbought condition for momentum

                (dataframe['atr'] > dataframe['atr'].rolling(window=14).mean()) |



                False  # Placeholder, replace with actual condition
            ),
            'sell'
        ] = 1

        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (

                (dataframe['volume'] > 0)

            ),

            'buy'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe.loc[
            (

                (dataframe['volume'] > 0)

            ),

            'sell'
        ] = 1

        return dataframe
    
    

    