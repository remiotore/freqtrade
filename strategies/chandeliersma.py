from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class ChandelierSMA_Oscar(IStrategy):
    
    # Define the minimal ROI (Return on Investment)
    minimal_roi = {
        "0": 0.1,  # 10% ROI at any time
    }

    # Stoploss configuration
    stoploss = -0.06  # 10% stoploss

    # Define the timeframe for the strategy
    timeframe = '15m'

    # Indicator parameters
    zl_sma_length = 40
    chandelier_multiplier = 3.0
    atr_length = 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators used in the strategy.
        """
        # Zero Lag SMA calculation
        ema1 = ta.EMA(dataframe['close'], timeperiod=self.zl_sma_length)
        ema2 = ta.EMA(ema1, timeperiod=self.zl_sma_length)
        dataframe['zl_sma'] = ema1 + (ema1 - ema2)

        # Average True Range (ATR) calculation
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.atr_length)
        
        # Chandelier Exit calculation
        dataframe['chandelier_exit'] = dataframe['high'].rolling(window=self.atr_length).max() - self.chandelier_multiplier * dataframe['atr']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate buy signal logic.
        """
        dataframe.loc[
            (dataframe['close'] > dataframe['zl_sma']) &
            (dataframe['close'] > dataframe['chandelier_exit']),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate sell signal logic.
        """
        dataframe.loc[
            (dataframe['close'] < dataframe['zl_sma']),
            'sell'] = 1
        return dataframe
