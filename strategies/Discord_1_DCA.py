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
from freqtrade.optimize.space import SKDecimal
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class DCA(IStrategy):

    # Nested Hyperopt stoploss
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.99, -0.15, decimals=2, name='stoploss')]

    INTERFACE_VERSION = 2

    DATESTAMP = 0
    COUNT = 1
    DCA_ENABLE = 2
    
    # ROI table:
    minimal_roi = {
        "0": 1000
    }

    # Stoploss:
    stoploss = -0.33

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0.001
    # trailing_stop_positive_offset = 0.01
    # trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False
    
    # Optimal timeframe for the strategy
    # timeframe = '5m'
    timeframe = '4h'
    process_only_new_candles = True
    startup_candle_count = 60

    plot_config = {

    }

    # DCA config
    position_adjustment_enable = True 
    max_dca_orders = 2  # n - 1
    max_dca_multiplier = 7 # (2^n - 1)
    dca_trigger = 0

    # storage dict for custom info
    custom_info = { }

    # Hyperoptable parameters
    # This is where I enter my parameters for HO

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #This is where I enter my indicators from TALIB

        # Check if the entry already exists
        pair = metadata['pair']
        if not pair in self.custom_info:
            # Create empty entry for this pair {DATESTAMP, COUNT, DCA_ENABLE}
            self.custom_info[pair] = ['', 7, 0] 
        
        count = self.custom_info[pair][self.COUNT]
        
        dataframe['buy_1'] = 0
        dataframe['sell_1'] = 0

        # iterate through dataframe, create buys and sells
        row = 0
        last_row = dataframe.tail(1).index.item()
        while (row <= last_row):
                        
            if(count ==0):
                dataframe['buy_1'].iloc[row] = 1
            if(count == 2): 
                dataframe['buy_1'].iloc[row] = 1
            if(count == 4):
                dataframe['buy_1'].iloc[row] = 1
            if (count == 6):
                dataframe['sell_1'].iloc[row] = 1
                
            if (count == 7):
                count = 0
            else:
                count += 1
            
            if(row == 0):
                self.custom_info[pair][self.COUNT] = count

            row += 1

        


        return dataframe


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        self.custom_info[pair][self.DCA_ENABLE] = 0
        return True

    
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
            # new candle, trigger only once per new candle
            self.custom_info[trade.pair][self.DATESTAMP] = last_candle['date']

            if(last_candle['buy_1'] == 1):
                dca_enable = self.custom_info[trade.pair][self.DCA_ENABLE]
                self.custom_info[trade.pair][self.DCA_ENABLE] = 1
            
                if(dca_enable):
                    filled_buys = trade.select_filled_orders('buy')
                    count_of_buys = len(filled_buys)
                    if 0 < count_of_buys <= self.max_dca_orders:
                        try:
                            # This returns first order stake size
                            stake_amount = filled_buys[0].cost
                            # This then calculates current safety order size
                            stake_amount = stake_amount * pow(0.9, count_of_buys)
                            # print("--------------------------------") 
                            # print(trade.open_date_utc)
                            # print("--------------------------------")
                            # print("count_of_buys = " + str(count_of_buys))
                            # print("stake_amount = " + str(stake_amount))
                            return stake_amount
                        except Exception as exception:
                            print("exception")
                            return None
                    return None
        return None
    

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['buy_1'] = 0
        dataframe.loc[
            (
                # These are my buy signals
            ),
                ['buy','buy_1']
            ]=(1, 1)
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sell_1'] = 0
        dataframe.loc[
            (
                # These are my sell signals
            ),
                ['sell','sell_1']
            ]=(1, 1)
        return dataframe
