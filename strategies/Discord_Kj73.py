from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class Kj73(IStrategy):

     INTERFACE_VERSION: int = 3

     # Enter short positions?
     can_short = True
     
     # ROI table:
     minimal_roi = {"0": 0.15, "30": 0.1, "60": 0.05}

     # Stoploss:
     stoploss = -0.265

     # Trailing stop:
     trailing_stop = True
     trailing_stop_positive = 0.05
     trailing_stop_positive_offset = 0.1
     trailing_only_offset_is_reached = False

     timeframe = "15m"

     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
         # Calculate the two exponential moving averages
         fast_ema = ta.EMA(dataframe, timeperiod=10)
         slow_ema = ta.EMA(dataframe, timeperiod=21)

         # Calculate the RSI indicator
         rsi = ta.RSI(dataframe, timeperiod=14)

         # Calculate the CDC Action Zone V3 indicator
         az_up = fast_ema > slow_ema
         az_dn = fast_ema < slow_ema
         az = az_up.astype(int) - az_dn.astype(int)
         az_diff = az.diff()
         cdc_az_v3 = (az_diff == -2).astype(int) * -1 + (az_diff == 2).astype(int)

         # Add the indicators to the dataframe
         dataframe['fast_ema'] = fast_ema
         dataframe['slow_ema'] = slow_ema
         dataframe['rsi'] = rsi
         dataframe['cdc_az_v3'] = cdc_az_v3

         return dataframe

     def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
         # Generate enter signals for long signals here
         dataframe.loc[
             (dataframe['cdc_az_v3'] == 1) &
             (dataframe['rsi'] < 30),
             'enter_long'] = 1

         # Generate enter signals for short positions here
         dataframe.loc[
             (dataframe['cdc_az_v3'] == -1) &
             (dataframe['rsi'] > 70),
             'enter_short'] = 1

         return dataframe

     def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
         # Generate exit signals for long positions here
         dataframe.loc[
             (dataframe['cdc_az_v3'] == -1) &
             (dataframe['rsi'] > 70),
             'exit_long'] = 1

         # Generate exit signals for short positions here
         dataframe.loc[
             (dataframe['cdc_az_v3'] == 1) &
             (dataframe['rsi'] < 30),
             'exit_short'] = 1

         return dataframe