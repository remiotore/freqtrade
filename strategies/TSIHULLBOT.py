


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import math









def hma(dataframe, source, length):
    return ta.WMA(
                    2 * ta.WMA(dataframe[source], int(math.floor(length/2))) - ta.WMA(dataframe[source], length), int(round(np.sqrt(length)))
            )

def double_smooth(dataframe, price, long, short):
    dft = dataframe.copy()
    dft['fist_smooth'] = hma(dataframe,  source = price, length = long)
    return hma(dft, source = 'fist_smooth', length = short)

 
class TSIHULLBOT(IStrategy):
    minimal_roi = { "0": -1 }

    buy_params = {
        "long_buy": 61,
        "price_buy": "close",
        "short_buy": 38,
        "signal_buy": 46,
    }

    sell_params = {
        "long_sell": 80,
        "price_sell": "open",
        "short_sell": 29,
        "signal_sell": 74,
    }

    long_buy = IntParameter(5, 80, default= int(buy_params['long_buy']), space='buy')
    short_buy = IntParameter(5, 80, default=int(buy_params['short_buy']), space='buy')
    signal_buy = IntParameter(1, 80, default=int(buy_params['signal_buy']), space='buy')
    price_buy = CategoricalParameter(['open', 'close'], default=str(buy_params['price_buy']), space='buy')

    long_sell = IntParameter(5, 80, default=int(sell_params['long_sell']), space='sell')
    short_sell = IntParameter(5, 80, default=int(sell_params['short_sell']), space='sell')
    signal_sell = IntParameter(1, 80, default=int(sell_params['signal_sell']), space='sell')
    price_sell = CategoricalParameter(['open', 'close'], default=str(sell_params['price_sell']), space='sell')
  
    INTERFACE_VERSION = 2

    stoploss = -0.99
    trailing_stop = False
    timeframe = '1h'
    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def informative_pairs(self):
       
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['pc_buy']                     =  dataframe[self.price_buy.value].diff()
        dataframe['pc_abs_buy']                 =  dataframe['pc_buy'].abs()
        dataframe['double_smoothed_pc_buy']     =  double_smooth(dataframe, 'pc_buy',     self.long_buy.value, self.short_buy.value)
        dataframe['double_smoothed_abs_pc_buy'] =  double_smooth(dataframe, 'pc_abs_buy', self.long_buy.value, self.short_buy.value)
        
        
        dataframe['tsi_value_buy']              =  ( 100 * (dataframe['double_smoothed_pc_buy']/dataframe['double_smoothed_abs_pc_buy'] )) * 5
        dataframe['tsihmaline_buy']             =  hma(dataframe,  source = 'tsi_value_buy', length = self.signal_buy.value) * 5

        dataframe['pc_sell']                     =  dataframe[self.price_sell.value].diff()
        dataframe['pc_abs_sell']                 =  dataframe['pc_sell'].abs()
        dataframe['double_smoothed_pc_sell']     =  double_smooth(dataframe, 'pc_sell',     self.long_sell.value, self.short_sell.value)
        dataframe['double_smoothed_abs_pc_sell'] =  double_smooth(dataframe, 'pc_abs_sell', self.long_sell.value, self.short_sell.value)
        
        
        dataframe['tsi_value_sell']              =  ( 100 * (dataframe['double_smoothed_pc_sell']/dataframe['double_smoothed_abs_pc_sell'] )) * 5
        dataframe['tsihmaline_sell']             =  hma(dataframe,  source = 'tsi_value_sell', length = self.signal_sell.value) * 5

        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
     
        dataframe.loc[
            (
                 
                (dataframe['tsihmaline_buy'] > dataframe['tsihmaline_buy'].shift(1)) &  
                (dataframe[self.price_buy.value] > dataframe[self.price_buy.value].shift(1)) &  
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
       
        dataframe.loc[
            (
                (dataframe['tsihmaline_sell'] < dataframe['tsihmaline_sell'].shift(1)) &  
                (dataframe[self.price_sell.value] < dataframe[self.price_sell.value].shift(1)) &  
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    