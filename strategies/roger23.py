import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime, timezone
# import logging  # remove after
# logger = logging.getLogger(__name__)  # remove after


class roger(IStrategy):

    INTERFACE_VERSION = 3

    # Short / Long
    can_short = False

    # ROI table (config.json can't have this value as it will override): 
    minimal_roi = {
    "7200": 0.01, # After 5 days, take 1% profit
    "4320": 0.03, # After 3 days, take 2% profit
    "2880": 0.04, # After 2 days, take 3% profit
    "1440": 0.05, # After 1 day, take 4% profit
    "480":  0.08, # After 8 hours, take 5% profit
    "120":  0.10, # After 2 hours, take 8% profit
    "15":  0.15, # After 15 minutes, take 10% profit
    "0":   0.20 
    }

    # Stoploss (config.json can't have this value as it will override)
    stoploss = -0.20
    
    # Custom Stoplos
    use_custom_stoploss = True
   
    # Trailing stoploss
    trailing_stop = False
    
    # Timeframe
    timeframe = '15m'
    
    # Run "populate_indicators" only for new candle
    process_only_new_candles = True
    
    max_profits = {}
    
    def on_trade_update(self, trade: Trade, **kwargs):
        # Update max_profit for the trade
        if trade.pair not in self.max_profits:
            self.max_profits[trade.pair] = 0
        self.max_profits[trade.pair] = max(self.max_profits[trade.pair], trade.current_profit_ratio)
        
    def on_trade_close(self, trade: Trade, **kwargs):
        # Remove max_profit for the trade
        self.max_profits.pop(trade.pair, None)
    
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
        
        # Calclate ATR
        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=150)
        # Calculate ATR stoploss
        dataframe['ATR_stoploss'] = dataframe['ATR'] * 6.5
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
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
            (dataframe['sma20'] > dataframe['sma50']) & # Short term trend indicator
            (dataframe['sma50'] > dataframe['sma200']) & # Long term trend indicator
            (dataframe['bb_percent'] < 0.2) & # Price is near the lower Bollinger Band
            (dataframe['bb_width'] < 0.05) # Bollinger Bands are narrow
        )
        
        # Candlestick patterns
        candlestick_patterns = (
            (dataframe['cdl3inside'] == 100) |  # 3 Inside Up
            (dataframe['cdl3outside'] == 100) |  # 3 Outside Up
            (dataframe['cdl3starsinsouth'] == 100) |  # 3 Stars In The South
            (dataframe['cdlhammer'] == 100) |  # Hammer
            (dataframe['cdlinvertedhammer'] == 100)  # Inverted Hammer
        )
        
        # MACD-based entry conditions
        macd_condition = (
            (dataframe['macd'] > dataframe['macdsignal'])  # MACD line above signal line
        )
        
        # Strong combinded buy signal
        dataframe.loc[bullish_conditions | candlestick_patterns & macd_condition, 'buy'] = 1
          
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
            (dataframe['bb_width'] > 0.03)
        )
        
        # Candlestick patterns
        candlestick_patterns = (
            (dataframe['cdl3blackcrows'] == -100) |  # 3 Black Crows
            (dataframe['cdl3whitesoldiers'] == 100) |  # 3 White Soldiers
            (dataframe['cdl3linestrike'] == -100) |  # 3 Line Strike
            (dataframe['cdlgravestonedoji'] == -100) |  # Gravestone Doji
            (dataframe['cdlshootingstar'] == -100)  # Shooting Star
        )
        
        # MACD-based exit conditions
        macd_condition = (
        (dataframe['macd'] < dataframe['macdsignal'])  # MACD line below signal line
    )
        
        # Apply sell signal
        dataframe.loc[bearish_conditions & candlestick_patterns & macd_condition, 'sell'] = 1
        
        return dataframe
    
    
    def calc_stop_loss_pct(self, current_rate: float, atr_multiplier: float) -> float:
        return -atr_multiplier * current_rate
        
    def custom_stoploss(self, pair: str, trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:
        # Retrieve max_profit for the current pair; default to current_profit if not found
        max_profit = self.max_profits.get(pair, current_profit)

        # Calculate the drawdown from the peak profit
        peak_profit_drawdown = max_profit - current_profit

        # Initial ATR-based stop loss adjustment
        atr_stoploss = self.calc_stop_loss_pct(current_rate, 6.5)

        # Adjust stop loss based on drawdown from peak profit
        if peak_profit_drawdown > 0.05:  # If drawdown from peak is greater than 5%
            atr_stoploss *= 0.8  # Tighten the stop loss by 20%
        elif peak_profit_drawdown > 0.1:  # If drawdown from peak is greater than 10%
            atr_stoploss *= 0.6  # Tighten the stop loss by 40%

        # Adjust stop loss based on current profit
        if current_profit > 0.05:  # If current profit is above 5%
            atr_stoploss *= 0.8  # Tighten the stop loss by 20%
        elif current_profit > 0.1:  # If current profit is above 10%
            atr_stoploss *= 0.6  # Tighten the stop loss by 40%

        # Ensure the custom stop loss is not looser than the initial stop loss
        adjusted_stoploss = max(atr_stoploss, self.stoploss)

        return adjusted_stoploss