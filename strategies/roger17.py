import numpy as np
import pandas as pd
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import IStrategy, stoploss_from_absolute, stoploss_from_open
from freqtrade.persistence import Trade
from datetime import datetime
import logging  # remove after
logger = logging.getLogger(__name__)  # remove after


class roger17(IStrategy):

    INTERFACE_VERSION = 3

    can_short = False

    minimal_roi = {
    "1440": 0.005, # After 24 hours, require at least a 0.5% ROI to sell
    "720": 0.01,  # After 12 hours, require at least a 1% ROI to sell
    "240": 0.02,  # After 4 hours, require at least a 2% ROI to sell
    "120": 0.025, # After 2 hours, require at least a 2.5% ROI to sell
    "60":  0.03,  # After 1 hour, require at least a 3% ROI to sell
    "30":  0.04,  # After 30 minutes, require at least a 4% ROI to sell
    "15":  0.05,  # After 15 minutes, require at least a 5% ROI to sell
    "0":   0.07   # After 0 minutes, require at least a 7% ROI to sell
    }

    stoploss = -0.20

    use_custom_stoploss = True

    trailing_stop = False

    timeframe = '15m'

    process_only_new_candles = True
    
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
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_profit: float, current_rate: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if trade.stop_loss == trade.initial_stop_loss:


            atr_stoploss = stoploss_from_absolute(last_candle['ATR_stoploss'], current_rate)
            if np.isnan(atr_stoploss):
                return None
            else:
                return atr_stoploss
            
        if current_profit > 0.01:


            divided_profit = current_profit / 2
            return stoploss_from_open(divided_profit, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        return trade.stop_loss
