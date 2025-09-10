from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter, CategoricalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime

class AdvancedFuturesStrategy(IStrategy):
    # Optimal timeframe for the strategy
    timeframe = '15m'

    # Minimal ROI (Return on Investment)
    minimal_roi = {
        "0": 0.20,  # 20% ROI for any trade
        "30": 0.10, # Reduce ROI to 10% after 30 minutes
        "60": 0.05, # Reduce ROI to 5% after 60 minutes
        "120": 0    # Exit after 120 minutes
    }

    # Stoploss
    stoploss = -0.10  # 10% stoploss

    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05

    # Dynamic position sizing based on volatility
    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    # Futures-specific settings
    use_custom_stoploss = True  # Enable custom stoploss logic
    leverage = IntParameter(1, 10, default=3, space='buy', optimize=True)  # Dynamic leverage
    funding_rate_threshold = DecimalParameter(-0.0005, 0.0005, default=-0.0001, space='buy', optimize=True)  # Avoid high funding costs

    # Define hyperparameters for optimization
    buy_rsi = IntParameter(20, 40, default=30, space='buy', optimize=True)
    sell_rsi = IntParameter(60, 80, default=70, space='sell', optimize=True)
    ema_short = IntParameter(5, 15, default=9, space='buy', optimize=True)
    ema_long = IntParameter(20, 50, default=21, space='buy', optimize=True)
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='buy', optimize=True)

    # Define indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI (Relative Strength Index)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # EMA (Exponential Moving Average)
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.ema_short.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.ema_long.value)

        # ATR (Average True Range) for volatility-based position sizing
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Bollinger Bands for mean-reversion
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']

        # MACD (Moving Average Convergence Divergence)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Funding rate (if available)
        if 'funding_rate' in dataframe:
            dataframe['funding_rate'] = dataframe['funding_rate']
        else:
            dataframe['funding_rate'] = 0  # Default to 0 if funding rate data is unavailable

        return dataframe

    # Define entry (buy) signals
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) &  # RSI below threshold (oversold)
                (dataframe['ema_short'] > dataframe['ema_long']) &  # Short EMA above Long EMA (uptrend)
                (dataframe['close'] < dataframe['bb_lower']) &  # Price below Bollinger Lower Band (mean-reversion)
                (dataframe['macdhist'] > 0) &  # MACD histogram positive (momentum)
                (dataframe['funding_rate'] > self.funding_rate_threshold.value)  # Avoid high negative funding rates
            ),
            'enter_long'] = 1
        return dataframe

    # Define exit (sell) signals
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value) |  # RSI above threshold (overbought)
                (dataframe['close'] > dataframe['bb_upper']) |  # Price above Bollinger Upper Band (mean-reversion)
                (dataframe['macdhist'] < 0) |  # MACD histogram negative (momentum loss)
                (dataframe['funding_rate'] < self.funding_rate_threshold.value)  # Avoid high positive funding rates
            ),
            'exit_long'] = 1
        return dataframe

    # Dynamic position sizing based on volatility
    def custom_stake_amount(self, pair: str, current_time, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        atr = last_candle['atr']
        atr_multiplier = self.atr_multiplier.value

        # Calculate stake size based on ATR (volatility)
        stake_size = max(min_stake, min(max_stake, proposed_stake * (atr_multiplier / atr)))
        return stake_size

    # Custom stoploss logic for futures
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        # Use a trailing stoploss with a dynamic buffer
        if current_profit > 0.05:  # Lock in profits after 5% gain
            return current_profit - 0.03  # Keep a 3% buffer
        return self.stoploss  # Default stoploss
