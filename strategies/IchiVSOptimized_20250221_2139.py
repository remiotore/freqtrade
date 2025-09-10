import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import stoploss
from pandas import DataFrame
from typing import Dict, List, Optional, Tuple

class OptimizedIchimokuStrategy(IStrategy):
    INTERFACE_VERSION = 3
    
    minimal_roi = {
        "0": 0.1,
        "30": 0.05,
        "60": 0.02,
        "120": 0
    }
    
    stoploss = -0.10
    
    timeframe = '5m'
    startup_candle_count: int = 100
    process_only_new_candles = True
    use_custom_stoploss = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Ichimoku Cloud
        nine_period_high = dataframe['high'].rolling(window=9).max()
        nine_period_low = dataframe['low'].rolling(window=9).min()
        dataframe['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        period26_high = dataframe['high'].rolling(window=26).max()
        period26_low = dataframe['low'].rolling(window=26).min()
        dataframe['kijun_sen'] = (period26_high + period26_low) / 2
        
        dataframe['senkou_span_a'] = ((dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2).shift(26)
        
        period52_high = dataframe['high'].rolling(window=52).max()
        period52_low = dataframe['low'].rolling(window=52).min()
        dataframe['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        dataframe['chikou_span'] = dataframe['close'].shift(-26)
        
        # EMA Filters
        dataframe['ema_50'] = dataframe['close'].ewm(span=50, adjust=False).mean()
        dataframe['ema_200'] = dataframe['close'].ewm(span=200, adjust=False).mean()
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] > dataframe['senkou_span_a']) &
            (dataframe['close'] > dataframe['senkou_span_b']) &
            (dataframe['tenkan_sen'] > dataframe['kijun_sen']) &
            (dataframe['ema_50'] > dataframe['ema_200']),
            'buy'] = 1
        return dataframe
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['senkou_span_a']) &
            (dataframe['close'] < dataframe['senkou_span_b']) &
            (dataframe['tenkan_sen'] < dataframe['kijun_sen']) &
            (dataframe['ema_50'] < dataframe['ema_200']),
            'sell'] = 1
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        if current_profit > 0.05:
            return -0.02
        return -0.10
