import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime
import logging  # remove after
logger = logging.getLogger(__name__)  # remove after


class roger(IStrategy):

    INTERFACE_VERSION = 3

    # Short / Long
    can_short = False

    # ROI table (config.json can't have this value as it will override): 
    minimal_roi = {
    "240": 0.02,  # After 4 hours, require at least a 2% ROI to sell
    "120": 0.025, # After 2 hours, require at least a 2.5% ROI to sell
    "60":  0.03,  # After 1 hour, require at least a 3% ROI to sell
    "30":  0.04,  # After 30 minutes, require at least a 4% ROI to sell
    "0":   0.05   # Initially, require at least a 5% ROI to sell
    }

    # Stoploss (config.json can't have this value as it will override)
    stoploss = -0.20

    # Trailing stoploss
    trailing_stop = False
    
    # Timeframe
    timeframe = '15m'
    
    # Run "populate_indicators" only for new candle
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populates new indicators for given strategy

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """
        
        # SMA
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = \
            (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        # Candlestick patterns bullish
        dataframe['cdl3inside'] = ta.CDL3INSIDE(dataframe)
        dataframe['cdl3outside'] = ta.CDL3OUTSIDE(dataframe)
        dataframe['cdl3starsinsouth'] = ta.CDL3STARSINSOUTH(dataframe)
        dataframe['cdlhammer'] = ta.CDLHAMMER(dataframe)
        dataframe['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(dataframe)
        
        # Candlestick patterns bearish
        dataframe['cdl3blackcrows'] = ta.CDL3BLACKCROWS(dataframe)
        dataframe['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(dataframe)
        dataframe['cdl3linestrike'] = ta.CDL3LINESTRIKE(dataframe)
        dataframe['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(dataframe)
        dataframe['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(dataframe)
        
        # Bullish or bearish
        dataframe['cdlengulfing'] = ta.CDLENGULFING(dataframe)
        
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "buy" signal

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """
        # Define bullish conditions based on SMA and Bollinger Bands
        bullish_conditions = (
            (dataframe['sma50'] > dataframe['sma200']) &
            (dataframe['bb_percent'] < 0.1) &
            (dataframe['bb_width'] < 0.03)
        )
        
        # Candlestick patterns
        candlestick_patterns = (
            (dataframe['cdl3inside'] == 100) |  # 3 Inside Up
            (dataframe['cdl3outside'] == 100) |  # 3 Outside Up
            (dataframe['cdl3starsinsouth'] == 100) |  # 3 Stars In The South
            (dataframe['cdlhammer'] == 100) |  # Hammer
            (dataframe['cdlinvertedhammer'] == 100)  # Inverted Hammer
        )
        
        # Apply buy signal based on combined bullish conditions and candlestick patterns
        dataframe.loc[bullish_conditions & candlestick_patterns, 'buy'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "sell" signal
        
        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair
            
        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """
        # Define bearish conditions based on SMA and Bollinger Bands
        bearish_conditions = (
            (dataframe['sma50'] < dataframe['sma200']) &
            (dataframe['bb_percent'] > 0.9) &
            (dataframe['bb_width'] > 0.1)
        )
        
        # Candlestick patterns
        candlestick_patterns = (
            (dataframe['cdl3blackcrows'] == -100) |  # 3 Black Crows
            (dataframe['cdl3whitesoldiers'] == 100) |  # 3 White Soldiers
            (dataframe['cdl3linestrike'] == -100) |  # 3 Line Strike
            (dataframe['cdlgravestonedoji'] == -100) |  # Gravestone Doji
            (dataframe['cdlshootingstar'] == -100)  # Shooting Star
        )
        
        # Dynamic stop-loss based on ATR
        atr_multiplier = 6
        stop_loss_condition = dataframe['close'] - (dataframe['atr'] * atr_multiplier) > dataframe['close'].shift()
        
        # Combine bearish conditions, candlestick patterns, and dynamic stop-loss condition
        sell_signal = bearish_conditions & candlestick_patterns | stop_loss_condition
        
        # Apply sell signal
        dataframe.loc[sell_signal, 'sell'] = 1
        
        return dataframe
