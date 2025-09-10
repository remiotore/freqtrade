import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime
import logging  # remove after
logger = logging.getLogger(__name__)  # remove after

class roger8(IStrategy):

    INTERFACE_VERSION = 3

    can_short = False

    """minimal_roi = {
        "120": 0.005,
        "60":  0.01, 
        "30":  0.03,
        "20":  0.04, 
        "0":  1
    }"""

    stoploss = -0.20

    trailing_stop = False

    timeframe = '15m'

    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populates new indicators for given strategy

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """

        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = \
            (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        dataframe['cdl3inside'] = ta.CDL3INSIDE(dataframe)
        dataframe['cdl3outside'] = ta.CDL3OUTSIDE(dataframe)
        dataframe['cdl3starsinsouth'] = ta.CDL3STARSINSOUTH(dataframe)



        dataframe['cdl3blackcrows'] = ta.CDL3BLACKCROWS(dataframe)
        dataframe['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(dataframe)
        dataframe['cdl3linestrike'] = ta.CDL3LINESTRIKE(dataframe)



        dataframe['cdlengulfing'] = ta.CDLENGULFING(dataframe)
        
     
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "buy" signal

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """

        dataframe.loc[
            (
                (dataframe['sma200'] < dataframe['sma50']) &
                (dataframe['bb_percent'] < 0.1) &
                (dataframe['bb_width'] > 0.03) |

                (dataframe['cdl3inside'] == 100) | # 3 Inside Up/Down
                (dataframe['cdl3outside'] == 100) | # 3 Outside Up/Down
                (dataframe['cdl3starsinsouth'] == 100) | # 3 Stars In The South
                (dataframe['cdlengulfing'] == 100) # Engulfing Pattern


            ),


            'buy'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "sell" signal
        
        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair
            
        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """

        dataframe.loc[
            (
                (dataframe['sma200'] > dataframe['sma50']) &
                (dataframe['bb_percent'] > 0.9) &
                (dataframe['bb_width'] > 0.03) |

                (dataframe['cdl3blackcrows'] == -100) | # 3 Black Crows
                (dataframe['cdl3whitesoldiers'] == -100) | # 3 White Soldiers
                (dataframe['cdl3linestrike'] == -100) | # 3 Line Strike


                (dataframe['cdlengulfing'] == -100) # Engulfing Pattern
            ),

            'sell'
        ] = 1
        
        return dataframe
