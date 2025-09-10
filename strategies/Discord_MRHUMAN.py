from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta


class MRHUMAN(IStrategy):

    # ROI table:
    minimal_roi = {
        "0": 0.21218,
        "38": 0.05886,
        "68": 0.023,
        "105": 0
    }

    # Stoploss:
    stoploss = -0.29203

     Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.13292
    trailing_stop_positive_offset = 0.2144
    trailing_only_offset_is_reached = False
  
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        #CCI
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=50)

        #RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)        
        
        # EMA
        dataframe['ema30'] = ta.EMA(dataframe, timeperiod=30)
        dataframe['ema60'] = ta.EMA(dataframe, timeperiod=60)
        dataframe['ema360'] = ta.EMA(dataframe, timeperiod=360)   

        # Smooth
        dataframe['rsi_smooth'] = ta.EMA(dataframe, timeperiod=5, price='rsi')   
     
        # Bollinger
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']

        # ADX 
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # +DM
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)

        # -DM
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
            	           	
            (    
                #Down
            	(dataframe['adx'] > 20) & (dataframe['plus_di'] < dataframe['minus_di']) &
            	(dataframe['macd'] < dataframe['macdsignal']) &
               (dataframe['cci'] < -28) &
                (dataframe['rsi_smooth'] < 32) &
                (dataframe['low'] < dataframe['bb_lowerband'])	)
                
             |    
                
            (    
                #Side
            	(dataframe['adx'] < 20) &
            	(dataframe['macd'] < dataframe['macdsignal']) &
               (dataframe['cci'] < 53) &
                (dataframe['rsi_smooth'] < 30) &
                (dataframe['low'] < dataframe['bb_lowerband'])	)


            |    
                
            (    
                #Up
            	(dataframe['adx'] > 20) & (dataframe['plus_di'] > dataframe['minus_di']) &
            	(dataframe['macd'] < dataframe['macdsignal']) &
               (dataframe['cci'] < 34) &
                (dataframe['rsi_smooth'] < 46) &
                (dataframe['low'] < dataframe['bb_lowerband'])	)


                
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
		     (   #Down
		     (dataframe['adx'] > 20) & (dataframe['plus_di'] < dataframe['minus_di']) &
		     (dataframe['macd'] > dataframe['macdsignal']) &
		     (dataframe['rsi_smooth'] > 72)	)

            |

            (   #Side
		     (dataframe['adx'] < 20) &
		     (dataframe['macd'] > dataframe['macdsignal']) &
		     (dataframe['rsi_smooth'] > 37)	)

            |

            (   #Up
		     (dataframe['adx'] > 20) & (dataframe['plus_di'] > dataframe['minus_di']) &
		     (dataframe['macd'] > dataframe['macdsignal']) &
		     (dataframe['rsi_smooth'] > 96)	)
		        
		     
            ),
            'sell'] = 1

        return dataframe
