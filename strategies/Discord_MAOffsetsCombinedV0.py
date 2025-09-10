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
from technical.indicators import zema

# Use it at your own risk, it wasn't tested dry/live yet, recommended to use limit buys/sells

MIN_CANDLES_BUY = 5
MAX_CANDLES_BUY = 55

def EWO(dataframe, ema_length=5, ema2_length=35):
	df = dataframe.copy()
	ema1 = ta.EMA(df, timeperiod=ema_length)
	ema2 = ta.EMA(df, timeperiod=ema2_length)
	emadif = (ema1 - ema2) / df['close'] * 100
	return emadif
		
def get_low_offset(val, base, inc):
	k = (val / 100) * inc
	return base + k
	
def get_high_offset(buy, offset):
	return buy + offset
    
class MAOffsetsCombinedV0(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "buy_1_base_offset": 0.975,
        "buy_1_condition": True, 
        "buy_1_ewo_high": 2.2,
        "buy_1_inc": -0.05, 
        "buy_1_max_nb_candles": 47, 
        "buy_1_min_nb_candles": 7, 
        "buy_1_rsi_fast": 39, 
        "buy_2_base_offset": 0.962,
        "buy_2_ewo_high": 0.3,
        "buy_2_inc": -0.02,
        "buy_2_max_nb_candles": 49,
        "buy_2_min_nb_candles": 12,
        "buy_2_rsi_fast": 42,
        "buy_2_condition": True, 
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_2_offset_inc": 0.021,
        "sell_1_offset_inc": 0.004,  
    }

    # ROI table:  # value loaded from strategy
    minimal_roi = {
        "0": 0.044,
        "10": 0.034,
        "35": 0.021,
        "48": 0.011,
        "132": 0
    }

    # Stoploss:
    stoploss = -0.237  # value loaded from strategy

    # Trailing stop:
    trailing_stop = True  # value loaded from strategy
    trailing_stop_positive = 0.005  # value loaded from strategy
    trailing_stop_positive_offset = 0.04  # value loaded from strategy
    trailing_only_offset_is_reached = True  # value loaded from strategy

    # Buy
  
    buy_1_condition = CategoricalParameter([True, False], default=buy_params['buy_1_condition'], space='buy', optimize=False, load=True)
    buy_1_min_nb_candles = IntParameter(5, 50, default=buy_params['buy_1_min_nb_candles'], space='buy', optimize=False)
    buy_1_max_nb_candles = IntParameter(5, 50, default=buy_params['buy_1_max_nb_candles'], space='buy', optimize=False)
    buy_1_base_offset = DecimalParameter(0.960, 0.990, default=buy_params['buy_1_base_offset'], space='buy', decimals=3, optimize=False)
    buy_1_inc = DecimalParameter(-0.24, 0.00, default=buy_params['buy_1_inc'], space='buy', decimals=2, optimize=False)
    buy_1_ewo_high = DecimalParameter(0.0, 6.0, default=buy_params['buy_1_ewo_high'], decimals=1, space='buy', optimize=False)
    buy_1_rsi_fast = IntParameter(5, 50, default=buy_params['buy_1_rsi_fast'], space='buy', optimize=False)
    
    buy_2_condition = CategoricalParameter([True, False], default=buy_params['buy_2_condition'], space='buy', optimize=False, load=True)
    buy_2_min_nb_candles = IntParameter(5, 50, default=buy_params['buy_2_min_nb_candles'], space='buy', optimize=True)
    buy_2_max_nb_candles = IntParameter(5, 50, default=buy_params['buy_2_max_nb_candles'], space='buy', optimize=True)
    buy_2_base_offset = DecimalParameter(0.960, 0.990, default=buy_params['buy_2_base_offset'], space='buy', decimals=3, optimize=True)
    buy_2_inc = DecimalParameter(-0.24, 0.00, default=buy_params['buy_2_inc'], space='buy', decimals=2, optimize=True)
    buy_2_ewo_high = DecimalParameter(0.0, 6.0, default=buy_params['buy_2_ewo_high'], decimals=1, space='buy', optimize=True)
    buy_2_rsi_fast = IntParameter(5, 50, default=buy_params['buy_2_rsi_fast'], space='buy', optimize=True)

    # Sell

    sell_1_offset_inc = DecimalParameter(0.001, 0.024, default=sell_params['sell_1_offset_inc'], space='sell', decimals=3, optimize=False)
    sell_2_offset_inc = DecimalParameter(0.001, 0.024, default=sell_params['sell_2_offset_inc'], space='sell', decimals=3, optimize=True)
    
    # Protection
    fast_ewo = 50
    slow_ewo = 200
    
    # Sell signal
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_get_low_offset = 0.01
    ignore_roi_if_buy_1_signal = True
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

    process_only_new_candles = True
    startup_candle_count = 500

    use_custom_stoploss = False
    
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)

        if (last_candle is not None):
        
        	if "trima" in trade.buy_tag:
        		
        		buy_offset = get_low_offset(int(trade.buy_tag.split('_')[-1]), self.buy_1_base_offset.value, self.buy_1_inc.value)
        		
        		sell_offset = get_high_offset(buy_offset, self.sell_1_offset_inc.value)
        		
        		if (last_candle['close'] > (last_candle[trade.buy_tag] * sell_offset)) & (last_candle['volume'] > 0):
        			return 'signal_sell_' + trade.buy_tag
        			
        	elif "ema" in trade.buy_tag:
        	
        		buy_offset = get_low_offset(int(trade.buy_tag.split('_')[-1]), self.buy_2_base_offset.value, self.buy_2_inc.value)
        		
        		sell_offset = get_high_offset(buy_offset, self.sell_2_offset_inc.value)
        		
        		if (last_candle['head'] > (last_candle[trade.buy_tag] * sell_offset)) & (last_candle['volume'] > 0):
        			return 'signal_sell_' + trade.buy_tag

        return None
    

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Bottoms and Ups
        dataframe['bottom'] = dataframe[['open', 'close']].min(axis=1)
        dataframe['top'] = dataframe[['open', 'close']].max(axis=1)
        dataframe['tail'] = (dataframe['bottom'] + dataframe['low']) / 2
        dataframe['head'] = (dataframe['top'] + dataframe['high']) / 2
        
        # Calculate all ma_buy values
        for val in range(MIN_CANDLES_BUY, MAX_CANDLES_BUY):
        	dataframe[f'trima_{val}'] = ta.TRIMA(dataframe, timeperiod=val)
        	
        for val in range(MIN_CANDLES_BUY, MAX_CANDLES_BUY):
        	dataframe[f'ema_{val}'] = ta.EMA(dataframe['tail'], timeperiod=val)
        		        
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        
        # RSI
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Adjust max candles
        buy_1_max_nb_candles = self.buy_1_max_nb_candles.value
        if buy_1_max_nb_candles <= self.buy_1_min_nb_candles.value:
        	buy_1_max_nb_candles = self.buy_1_min_nb_candles.value + 1
        	
        # Generate buy conditions
        for val in range(self.buy_1_min_nb_candles.value, buy_1_max_nb_candles):
        	tag = 'trima_' + str(val)
        	dataframe.loc[
        	(
                	(self.buy_1_condition.value == True) & 
                	
                	(dataframe['close'] < (dataframe[f'trima_{val}'] * get_low_offset(val, self.buy_1_base_offset.value, self.buy_1_inc.value))) & 
                	
                	(dataframe['EWO'] > self.buy_1_ewo_high.value) & 
                	
                	(dataframe['rsi_fast'] < self.buy_1_rsi_fast.value) &               	
                	
                	(dataframe['volume'] > 0)
        	),
        	['buy', 'buy_tag']] = (1, tag)
        	
        # Adjust max candles
        buy_2_max_nb_candles = self.buy_2_max_nb_candles.value
        if buy_2_max_nb_candles <= self.buy_2_min_nb_candles.value:
        	buy_2_max_nb_candles = self.buy_2_min_nb_candles.value + 1
        	
        # Generate buy conditions
        for val in range(self.buy_2_min_nb_candles.value, buy_2_max_nb_candles):
        	tag = 'ema_' + str(val)
        	dataframe.loc[
        	(
                	(self.buy_2_condition.value == True) & 
                	
                	(dataframe['tail'] < (dataframe[f'ema_{val}'] * get_low_offset(val, self.buy_2_base_offset.value, self.buy_2_inc.value))) & 
                	
                	(dataframe['EWO'] > self.buy_2_ewo_high.value) & 
                	
                	(dataframe['rsi_fast'] < self.buy_2_rsi_fast.value) &               	
                	
                	(dataframe['volume'] > 0)
        	),
        	['buy', 'buy_tag']] = (1, tag)
        	
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        dataframe.loc[:,'sell'] = 0

        return dataframe
