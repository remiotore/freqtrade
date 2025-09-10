import numpy as np
import pandas as pd
import pandas_ta as ta
import talib.abstract as talib
from technical import qtpylib
from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime
import logging  # remove after
logger = logging.getLogger(__name__)  # remove after


class roger2(IStrategy):

    INTERFACE_VERSION = 2

    timeframe = '15m'

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.2

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['ema200'] = talib.EMA(dataframe, timeperiod=200)

        dataframe['ema50'] = talib.EMA(dataframe, timeperiod = 50)

        return dataframe

   
    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            
            dataframe.loc[
                (

                    (dataframe['ema200'] > dataframe['ema50'])
                ),
                'buy'
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