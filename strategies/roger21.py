import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime, timezone




class roger21(IStrategy):

    INTERFACE_VERSION = 3

    can_short = False

    minimal_roi = {
    "1440": 0.005, # After 24 hours, require at least a 0.5% ROI to sell
    "720": 0.01,  # After 12 hours, require at least a 1% ROI to sell
    "480": 0.02, # After 8 hours, require at least 2% ROI to sell
    "240": 0.03,  # After 4 hours, require at least a 3% ROI to sell
    "120": 0.04, # After 2 hours, require at least a 4% ROI to sell
    "60":  0.05,  # After 1 hour, require at least a 5% ROI to sell
    "30":  0.06,  # After 30 minutes, require at least a 6% ROI to sell
    "15":  0.07,  # After 15 minutes, require at least a 7% ROI to sell
    "0":   0.10   # After 0 minutes, require at least a 10% ROI to sell
    }

    stoploss = -0.20

    use_custom_stoploss = True

    trailing_stop = False

    timeframe = '15m'

    process_only_new_candles = True
    
    max_profits = {}
    
    def on_trade_update(self, trade: Trade, **kwargs):

        if trade.pair not in self.max_profits:
            self.max_profits[trade.pair] = 0
        self.max_profits[trade.pair] = max(self.max_profits[trade.pair], trade.current_profit_ratio)
        
    def on_trade_close(self, trade: Trade, **kwargs):

        self.max_profits.pop(trade.pair, None)
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populates new indicators for given strategy

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """

        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = \
            (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        dataframe['cdl3inside'] = ta.CDL3INSIDE(dataframe)
        dataframe['cdl3outside'] = ta.CDL3OUTSIDE(dataframe)
        dataframe['cdl3starsinsouth'] = ta.CDL3STARSINSOUTH(dataframe)
        dataframe['cdlhammer'] = ta.CDLHAMMER(dataframe)
        dataframe['cdlinvertedhammer'] = ta.CDLINVERTEDHAMMER(dataframe)

        dataframe['cdl3blackcrows'] = ta.CDL3BLACKCROWS(dataframe)
        dataframe['cdl3whitesoldiers'] = ta.CDL3WHITESOLDIERS(dataframe)
        dataframe['cdl3linestrike'] = ta.CDL3LINESTRIKE(dataframe)
        dataframe['cdlgravestonedoji'] = ta.CDLGRAVESTONEDOJI(dataframe)
        dataframe['cdlshootingstar'] = ta.CDLSHOOTINGSTAR(dataframe)

        dataframe['cdlengulfing'] = ta.CDLENGULFING(dataframe)

        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=150)

        dataframe['ATR_stoploss'] = dataframe['ATR'] * 6.5
        
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "buy" signal

        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair

        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """

        bullish_conditions = (
            (dataframe['sma20'] > dataframe['sma50']) & # Short term trend indicator
            (dataframe['sma50'] > dataframe['sma200']) & # Long term trend indicator
            (dataframe['bb_percent'] < 0.2) & # Price is near the lower Bollinger Band
            (dataframe['bb_width'] < 0.05) # Bollinger Bands are narrow
        )

        candlestick_patterns = (
            (dataframe['cdl3inside'] == 100) |  # 3 Inside Up
            (dataframe['cdl3outside'] == 100) |  # 3 Outside Up
            (dataframe['cdl3starsinsouth'] == 100) |  # 3 Stars In The South
            (dataframe['cdlhammer'] == 100) |  # Hammer
            (dataframe['cdlinvertedhammer'] == 100)  # Inverted Hammer
        )

        dataframe.loc[bullish_conditions | candlestick_patterns, 'buy'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """ Populate rules for the "sell" signal
        
        Args:
            dataframe (pd.DataFrame): dataframe for the given pair
            metadata (dict): metadata for the given pair
            
        Returns:
            pd.DataFrame: dataframe with the defined indicators
        """

        bearish_conditions = (
            (dataframe['sma50'] < dataframe['sma200']) &
            (dataframe['bb_percent'] > 0.9) &
            (dataframe['bb_width'] > 0.03)
        )

        candlestick_patterns = (
            (dataframe['cdl3blackcrows'] == -100) |  # 3 Black Crows
            (dataframe['cdl3whitesoldiers'] == 100) |  # 3 White Soldiers
            (dataframe['cdl3linestrike'] == -100) |  # 3 Line Strike
            (dataframe['cdlgravestonedoji'] == -100) |  # Gravestone Doji
            (dataframe['cdlshootingstar'] == -100)  # Shooting Star
        )

        sell_signal = bearish_conditions | candlestick_patterns

        dataframe.loc[sell_signal, 'sell'] = 1
        
        return dataframe
    
    
    def calc_stop_loss_pct(self, current_rate: float, atr_multiplier: float) -> float:
        return -atr_multiplier * current_rate
        
    def custom_stoploss(self, pair: str, trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:

        max_profit = self.max_profits.get(pair, current_profit)

        peak_profit_drawdown = max_profit - current_profit

        atr_stoploss = self.calc_stop_loss_pct(current_rate, 6.5)

        if peak_profit_drawdown > 0.05:  # If drawdown from peak is greater than 5%
            atr_stoploss *= 0.8  # Tighten the stop loss by 20%
        elif peak_profit_drawdown > 0.1:  # If drawdown from peak is greater than 10%
            atr_stoploss *= 0.6  # Tighten the stop loss by 40%

        if current_profit > 0.05:  # If current profit is above 5%
            atr_stoploss *= 0.8  # Tighten the stop loss by 20%
        elif current_profit > 0.1:  # If current profit is above 10%
            atr_stoploss *= 0.6  # Tighten the stop loss by 40%

        adjusted_stoploss = max(atr_stoploss, self.stoploss)

        return adjusted_stoploss