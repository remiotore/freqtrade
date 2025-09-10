

"""
Created on Wed Dec  2 13:50:49 2020

@author: alex
"""

from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class tema_master_v2(IStrategy):
    """
    Sample strategy implementing Informative Pairs - compares stake_currency with USDT.
    Not performing very well - but should serve as an example how to use a referential pair against USDT.
    author@: xmatthias
    github@: https://github.com/freqtrade/freqtrade-strategies
    How to use it?
    > python3 freqtrade -s InformativeSample
    """



    minimal_roi = {
        "0": 0.31574,
        "367": 0.09547,
        "848": 0.04103,
        "1375": 0
    }



    stoploss = -0.11746

    timeframe = '1m'

    trailing_stop = True
    trailing_stop_positive = 0.17017
    trailing_stop_positive_offset = 0.26713
    trailing_only_offset_is_reached = True

    ta_on_candle = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False


    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [(f"{self.config['stake_currency']}/USDT", self.timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        dataframe['CMO'] = ta.CMO(dataframe, timeperiod = 180)
        dataframe['TEMA'] = ta.TEMA(dataframe, timeperiod = 60)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=60, stds=1.4)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
            
            
        dataframe['STDDEV'] = ta.STDDEV(dataframe, timeperiod=26, nbdev=1.4)
        dataframe['MA'] = ta.MA(dataframe, timeperiod=26, matype=0)
        dataframe["COEFFV"] = dataframe['STDDEV']/dataframe['MA']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
                        
        
        dataframe.loc[
            (
                  
            (
                (qtpylib.crossed_above(dataframe["TEMA"], dataframe["bb_lowerband"]))
            & 
              (dataframe['CMO']>0)
             )
                
                
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
                       
        dataframe.loc[
            (
               (qtpylib.crossed_below(dataframe["CMO"],-17))
                |
                (qtpylib.crossed_below(dataframe["CMO"],23)) 
            ),
            'sell'] = 1        
        
        return dataframe