# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from datetime import datetime


# --------------------------------


class BandRSI6(IStrategy):
    """
    author@: Gert Wohlgemuth
    Adapted by Radu Ulea & Giuseppe terranova
    --> essayer avec une sortie quand RSI  redeviens négatif
    
    
    converted from:
    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs
    """

    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 0.15
    }
    
   # Buy and sell at market price
    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }


    # Optimal stoploss designed for the strategy
    stoploss = -0.2
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe
        
        use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if ((current_profit < -0.099) and (dataframe['rsi'] < 30)):
    
            return 0 # return a value bigger than the inital stoploss to keep using the inital stoploss

        return 1
        
        

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (      (dataframe['close'] < dataframe['bb_lowerband']) #pour N
                       | (dataframe['close'].shift(1) < dataframe['bb_lowerband'].shift(1)) #pour N-1
                       | (dataframe['close'].shift(2) < dataframe['bb_lowerband'].shift(2)) #pour N-2
                       | (dataframe['close'].shift(3) < dataframe['bb_lowerband'].shift(3)) #pour N-3
                       | (dataframe['close'].shift(4) < dataframe['bb_lowerband'].shift(4)) #pour N-4
                       
               )
               
               &
               
               (
                              (dataframe['rsi'] > 30) &
                              (dataframe['rsi'].shift(1) <= 30)
               )


            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                   


            ),
            'sell'] = 1
        return dataframe
