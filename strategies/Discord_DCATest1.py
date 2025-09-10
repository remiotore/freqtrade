# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from freqtrade.exchange import timeframe_to_prev_date
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class DCATest1(IStrategy):
    INTERFACE_VERSION = 2

    DATESTAMP = 0
    COUNT = 1
    
    # ROI table:
    minimal_roi = {
        "0": 100.0
    }

    # Stoploss:
    stoploss = -0.99

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    
    # Optimal timeframe for the strategy
    # timeframe = '5m'
    timeframe = '1m'
    process_only_new_candles = True
    startup_candle_count = 0

    plot_config = {
        'subplots': {
            "first_buy": {
                'first_buy': {'color': 'yellow'}
            },
            "dca_buy": {
                'dca_buy': {'color': 'yellow'}
            },
            "all_sell": {
                'all_sell': {'color': 'yellow'}
            },
        }
    }

    # DCA config
    position_adjustment_enable = True 
    max_dca_orders = 2  # n - 1
    max_dca_multiplier = 7 # (2^n - 1)
    dca_trigger = 0

    # storage dict for custom info
    custom_info = { }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Check if the entry already exists
        pair = metadata['pair']
        if not pair in self.custom_info:
            # Create empty entry for this pair {DATESTAMP, COUNT}
            self.custom_info[pair] = ['', 13] 
        
        count = self.custom_info[pair][self.COUNT]
        
        dataframe['first_buy'] = 0
        dataframe['dca_buy'] = 0
        dataframe['all_sell'] = 0

        # iterate through dataframe, create buys and sells
        row = 0
        last_row = dataframe.tail(1).index.item()
        while (row <= last_row):
                        
            if(count ==0):
                dataframe['first_buy'].iloc[row] = 1
            if(count == 4): 
                dataframe['dca_buy'].iloc[row] = 1
            if(count == 8):
                dataframe['dca_buy'].iloc[row] = 1
            if (count == 12):
                dataframe['all_sell'].iloc[row] = 1
                
            if (count == 15):
                count = 0
            else:
                count += 1
            
            if(row == 0):
                self.custom_info[pair][self.COUNT] = count

            row += 1


        return dataframe


    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        return proposed_stake / self.max_dca_multiplier
  

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if(len(dataframe) < 1):
            return None
        last_candle = dataframe.iloc[-1].squeeze()

        if(self.custom_info[trade.pair][self.DATESTAMP] != last_candle['date']):
        # if(True):
            # new candle, trigger only once per new candle
            self.custom_info[trade.pair][self.DATESTAMP] = last_candle['date']
    
            if(last_candle['dca_buy'] == 1):
            # if((last_candle['dca_buy'] == 1) or (last_candle['first_buy'] == 1)):
                filled_buys = trade.select_filled_orders('buy')
                count_of_buys = len(filled_buys)
                if 0 < count_of_buys <= self.max_dca_orders:
                    try:
                        # This returns first order stake size
                        stake_amount = filled_buys[0].cost
                        # This then calculates current safety order size
                        stake_amount = stake_amount * pow(2, count_of_buys)
                        print("--------------------------------") 
                        print("count_of_buys = " + str(count_of_buys))
                        print("stake_amount = " + str(stake_amount))
                        return stake_amount
                    except Exception as exception:
                        print("exception")
                        return None
                return None
        return None
    

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'buy'] = 0
        dataframe.loc[
                # ((dataframe['first_buy'] == 1) | (dataframe['dca_buy'] == 1)),
                (dataframe['first_buy'] == 1),
                'buy'
            ]=1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0
        dataframe.loc[
                (dataframe['all_sell'] == 1),
                'sell'
            ]=1
        return dataframe
