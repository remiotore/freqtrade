# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
# --------------------------------

import freqtrade.vendor.qtpylib.indicators as qtpylib
from talipp.indicators import ALMA
from talipp.ohlcv import OHLCVFactory



class stoch(IStrategy):

	minimal_roi = {
          "0": 1000,
        
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
	stoploss = -0.05


    # Optimal timeframe for the strategy
	timeframe = '1h'
	
	startup_candle_count = 60
	
	

	def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
	
          high=dataframe['high']
          low=dataframe['low']
          close=dataframe['close']
          open=dataframe['open']
          
          dataframe['slowk'], dataframe['slowd'] = talib.STOCH(dataframe['high'], dataframe['low'],
           dataframe['close'],fastk_period=15,  slowk_period=17,slowk_matype=0,slowd_period=12,slowd_matype=0)

          
          return dataframe

	def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        	dataframe.loc[
            (

		     (qtpylib.crossed_above(
                dataframe['slowk'], dataframe['slowd']))


            ),
            'buy'] = 1

        	return dataframe

	def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        	dataframe.loc[
            (
            #   (qtpylib.crossed_above(
                # dataframe['slowd'], dataframe['slowk']))&
		      (dataframe['slowd']>95)&
              (dataframe['slowk']>69)
              

		      
		     
		          
            ),
            'sell'] = 1
        	return dataframe
        	
