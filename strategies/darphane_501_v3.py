from datetime import datetime
from typing import Optional
import logging
import talib.abstract as ta

import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, DecimalParameter, IntParameter, IStrategy,
                                RealParameter,informative,stoploss_from_open,timeframe_to_minutes,stoploss_from_absolute)
from functools import reduce
import time

from pandas import DataFrame, Series, DatetimeIndex, merge


from technical.util import resample_to_interval, resampled_merge
log = logging.getLogger(__name__)






class darphane_501_v3(IStrategy):
    INTERFACE_VERSION = 3
    def version(self) -> str:
        return "v106.00.00"
    
    minimal_roi = {
        "0": 100.00  
    }
    timeframe = '5m'
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc',
    }
    stoploss = -0.99

    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    use_custom_stoploss = True
    
    process_only_new_candles = True
    position_adjustment_enable = True
    
    can_short = True
    bot_started = True

    max_entry_position_adjustment = 5
    custom_info = {}












    @property
    def protections(self):
        return [
            
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 5,
                "trade_limit": 2,
                "stop_duration_candles": 2,
                "max_allowed_drawdown": 0.19
            }
        ]

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        filled_entries = trade.select_filled_orders(trade.entry_side)



        lvl = self.custom_info[f'{pair}']['Level']
        

        if (self.custom_info[f'{pair}']['Level'] == 1):
            if 2.00 < current_profit :
                return stoploss_from_open(2.00, current_profit, is_short=trade.is_short)
            elif 1.80 < current_profit <= 2.00: # 180%
                return stoploss_from_open(1.60, current_profit, is_short=trade.is_short)
            elif 1.70 < current_profit <= 1.80: # 170%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.60 < current_profit <= 1.80: # 160%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.50 < current_profit <= 1.80: # 150%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.40 < current_profit <= 1.50: # 140%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.30 < current_profit <= 1.50: # 130%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.00 < current_profit <= 1.30: # 100
                return stoploss_from_open(0.65, current_profit, is_short=trade.is_short)
            elif 0.90 < current_profit <= 1.00: # 90%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.80 < current_profit <= 0.90: # 80%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.70 < current_profit <= 0.80: # 70%
                return stoploss_from_open(0.40, current_profit, is_short=trade.is_short)
            elif 0.60 < current_profit <= 0.70: # 60%
                return stoploss_from_open(0.35, current_profit, is_short=trade.is_short)
            elif 0.40 < current_profit <= 0.60: # 40%
                return stoploss_from_open(0.30, current_profit, is_short=trade.is_short)
            elif 0.20 < current_profit <= 0.40: # 30%
                return stoploss_from_open(0.20, current_profit, is_short=trade.is_short)

        elif (self.custom_info[f'{pair}']['Level'] == 2):
            if 2.00 < current_profit :
                return stoploss_from_open(2.00, current_profit, is_short=trade.is_short)
            elif 1.80 < current_profit <= 2.00: # 180%
                return stoploss_from_open(1.60, current_profit, is_short=trade.is_short)
            elif 1.70 < current_profit <= 1.80: # 170%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.60 < current_profit <= 1.80: # 160%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.50 < current_profit <= 1.80: # 150%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.40 < current_profit <= 1.50: # 140%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.30 < current_profit <= 1.50: # 130%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.00 < current_profit <= 1.30: # 100
                return stoploss_from_open(0.65, current_profit, is_short=trade.is_short)
            elif 0.90 < current_profit <= 1.00: # 90%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.80 < current_profit <= 0.90: # 80%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.70 < current_profit <= 0.80: # 70%
                return stoploss_from_open(0.40, current_profit, is_short=trade.is_short)
            elif 0.60 < current_profit <= 0.70: # 60%
                return stoploss_from_open(0.35, current_profit, is_short=trade.is_short)
            elif 0.40 < current_profit <= 0.60: # 40%
                return stoploss_from_open(0.30, current_profit, is_short=trade.is_short)
            elif 0.20 < current_profit <= 0.40: # 30%
                return stoploss_from_open(0.20, current_profit, is_short=trade.is_short)

        elif (self.custom_info[f'{pair}']['Level'] == 3):
            if 2.00 < current_profit :
                return stoploss_from_open(2.00, current_profit, is_short=trade.is_short)
            elif 1.80 < current_profit <= 2.00: # 180%
                return stoploss_from_open(1.60, current_profit, is_short=trade.is_short)
            elif 1.70 < current_profit <= 1.80: # 170%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.60 < current_profit <= 1.80: # 160%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.50 < current_profit <= 1.80: # 150%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.40 < current_profit <= 1.50: # 140%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.30 < current_profit <= 1.50: # 130%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.00 < current_profit <= 1.30: # 100
                return stoploss_from_open(0.65, current_profit, is_short=trade.is_short)
            elif 0.90 < current_profit <= 1.00: # 90%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.80 < current_profit <= 0.90: # 80%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.70 < current_profit <= 0.80: # 70%
                return stoploss_from_open(0.40, current_profit, is_short=trade.is_short)
            elif 0.60 < current_profit <= 0.70: # 60%
                return stoploss_from_open(0.35, current_profit, is_short=trade.is_short)
            elif 0.40 < current_profit <= 0.60: # 40%
                return stoploss_from_open(0.30, current_profit, is_short=trade.is_short)
            elif 0.20 < current_profit <= 0.40: # 30%
                return stoploss_from_open(0.20, current_profit, is_short=trade.is_short)

        elif (self.custom_info[f'{pair}']['Level'] == 4):
            if 2.00 < current_profit :
                return stoploss_from_open(2.00, current_profit, is_short=trade.is_short)
            elif 1.80 < current_profit <= 2.00: # 180%
                return stoploss_from_open(1.60, current_profit, is_short=trade.is_short)
            elif 1.70 < current_profit <= 1.80: # 170%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.60 < current_profit <= 1.80: # 160%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.50 < current_profit <= 1.80: # 150%
                return stoploss_from_open(1.20, current_profit, is_short=trade.is_short)
            elif 1.40 < current_profit <= 1.50: # 140%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.30 < current_profit <= 1.50: # 130%
                return stoploss_from_open(1.00, current_profit, is_short=trade.is_short)
            elif 1.00 < current_profit <= 1.30: # 100
                return stoploss_from_open(0.65, current_profit, is_short=trade.is_short)
            elif 0.90 < current_profit <= 1.00: # 90%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.80 < current_profit <= 0.90: # 80%
                return stoploss_from_open(0.50, current_profit, is_short=trade.is_short)
            elif 0.70 < current_profit <= 0.80: # 70%
                return stoploss_from_open(0.40, current_profit, is_short=trade.is_short)
            elif 0.60 < current_profit <= 0.70: # 60%
                return stoploss_from_open(0.35, current_profit, is_short=trade.is_short)
            elif 0.40 < current_profit <= 0.60: # 40%
                return stoploss_from_open(0.30, current_profit, is_short=trade.is_short)
            elif 0.20 < current_profit <= 0.40: # 30%
                return stoploss_from_open(0.20, current_profit, is_short=trade.is_short)
            elif 0.10 < current_profit <= 0.20: # 30%
                return stoploss_from_open(0.10, current_profit, is_short=trade.is_short)

        
        return 1
        

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):


        if (self.custom_info[f'{pair}']['Level'] == 1):
            if (current_profit >= 0.10):

               return 'Level 1 Exit - Take Profit'
            if (current_profit <= -0.10):
                self.custom_info[f'{pair}']['Level'] = 2
                return 'Level 1 Exit - StopLoss / Entering Level 2'

        if (self.custom_info[f'{pair}']['Level'] == 2):
            if (current_profit >= 0.10):

               return 'Level 2 Exit - Take Profit'
            if (current_profit <= -0.10):
                self.custom_info[f'{pair}']['Level'] = 3
                return 'Level 2 Exit - StopLoss / Entering Level 3'

        if (self.custom_info[f'{pair}']['Level'] == 3):
            if (current_profit >= 0.10):

               return 'Level 3 Exit - Take Profit'
            if (current_profit <= -0.10):
                self.custom_info[f'{pair}']['Level'] = 4
                return 'Level 3 Exit - StopLoss / Entering Level 4'

        if (self.custom_info[f'{pair}']['Level'] == 4):
            if (current_profit >= 0.10):

               return 'Level 4 Exit - Take Profit'
            if (current_profit <= -0.10):
                self.custom_info[f'{pair}']['Level'] = 5
                return 'Level 4 Exit - StopLoss / Entering Level 5'

        if (self.custom_info[f'{pair}']['Level'] == 5):
            if (current_profit >= 0.10):

               return 'Level 5 Exit - Take Profit'
            if (current_profit <= -0.10):
                self.custom_info[f'{pair}']['Level'] = 6
                return 'Level 5 Exit - StopLoss / Entering Level 6'

        if (self.custom_info[f'{pair}']['Level'] == 6):
            if (current_profit >= 0.10):

               return 'Level 6 Exit - Take Profit'
            if (current_profit <= -0.10):
                self.custom_info[f'{pair}']['Level'] = 0
                return 'Level 6 Exit - StopLoss / RESET'
















        
        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:

        if (self.custom_info[pair]['Level'] == 2):

            proposed_stake = 30
        elif (self.custom_info[pair]['Level'] == 3):

            proposed_stake = 90
        elif (self.custom_info[pair]['Level'] == 4):

            proposed_stake = 270
        elif (self.custom_info[pair]['Level'] == 5):

            proposed_stake = 810
        elif (self.custom_info[pair]['Level'] == 6):

            proposed_stake = 1500







        return proposed_stake 

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
       
        return None

    def bot_start(self):
        self.bot_started = True

    def informative_pairs(self):

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata["pair"] in self.custom_info:

            self.custom_info[metadata["pair"]] = {}
            self.custom_info[metadata["pair"]]['Level'] = {}










        dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100, price='close')
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50, price='close')
        dataframe['sma25'] = ta.SMA(dataframe, timeperiod=25, price='close')
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200, price='close')
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cci_one'] = ta.CCI(dataframe, timeperiod=170)
        dataframe['cci_two'] = ta.CCI(dataframe, timeperiod=34)
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['cmf'] = chaikin_mf(dataframe) # period = 20
        dataframe['cmo'] = ta.CMO(dataframe) # period = 14





        str = f'{metadata["pair"]}'
        str = str.replace('/','_')
        dataframe.to_csv(f'user_data/pair_test/{str}_dataframe.csv', index=False, header=True, mode='w', encoding='UTF-8')
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                    (dataframe['cci_one'] < -100)
                    & (dataframe['cci_two'] < -100)
                    & (dataframe['cmf'] < -0.1)
                    & (dataframe['mfi'] < 25)
                    & (dataframe['rsi'] < 30)




            ),
            ['enter_long','enter_tag']] = (1,'Long1')
        
        dataframe.loc[
            (
                    (dataframe['cmo'] < -50)
                    & (dataframe['rsi'] < 22)
                    

            ),
            ['enter_long','enter_tag']] = (1,'Long2')
        
        
        
        dataframe.loc[
            (


                (dataframe['cci_one'] > 100)
                & (dataframe['cci_two'] > 100)
                & (dataframe['cmf'] > 0.1)
                & (dataframe['mfi'] > 75)
                & (dataframe['rsi'] > 75)
            ),
            ['enter_short','enter_tag']] = (1,'Short1')
            
        dataframe.loc[
            (
                (dataframe['cmo'] > 70)
              & (dataframe['rsi'] > 75)
            ),
            ['enter_short','enter_tag']] = (1,'Short2')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:

        return 1.0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        
        if (self.custom_info[f'{pair}']['Level'] == {} or self.custom_info[f'{pair}']['Level'] == 0):
            self.custom_info[f'{pair}']['Level'] = 1
        
        ll = self.custom_info[f'{pair}']['Level']

        return True
    
    def colored(self,param, color):
        color_codes = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
        }
        return color_codes[color] + param + '\033[0m'

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        filled_entries = trade.select_filled_orders(trade.entry_side)
        min_profit_rate = trade.max_rate if trade.is_short else trade.min_rate
        max_profit_rate = trade.min_rate if trade.is_short else trade.max_rate

        min_profit = (trade.calc_profit_ratio(min_profit_rate) * 100)
        max_profit = (trade.calc_profit_ratio(max_profit_rate) * 100)

        profit = (trade.calc_profit_ratio(rate) * 100)

        ll = self.custom_info[f'{pair}']['Level']
        
        if profit < 0:
            if ll == '0' or ll == 0 :
                log.info(
                f'EXIT Pair: {pair:13} | Level: {ll} | Exit Reason: {exit_reason} | Min_profit: {min_profit:.2f}% | Max_profit: {max_profit:.2f}% |  profit: {profit:.2f}% | {current_time}')
            else:
                self.log(
                f'EXIT Pair: {pair:13} | Level: {ll} | Exit Reason: {exit_reason} | Min_profit: {min_profit:.2f}% | Max_profit: {max_profit:.2f}% |  profit: {profit:.2f}% | {current_time}',color='red')
        else:
            self.log(
                f'EXIT Pair: {pair:13} | Level: {ll} | Exit Reason: {exit_reason} | Min_profit: {min_profit:.2f}% | Max_profit: {max_profit:.2f}% |  profit: {profit:.2f}% | {current_time}',color='green')
            self.custom_info[f'{pair}']['Level'] = 0
        
        return True

    def log(self, param, color):
        log.info(self.colored(param, color))

def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']

    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()

    return Series(cmf, name='cmf')



