from freqtrade.strategy import IStrategy
from freqtrade.util import DecimalParam, IntParam
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class EnhancedLimitedEntryDCA(IStrategy):
    """
    An enhanced version of the LimitedEntryDCA strategy with dynamic DCA parameters,
    additional entry signals, and advanced risk management.
    """

    # Strategy parameters
    # Initial entry parameters
    rsi_period = IntParam(14, default=14, space='buy', optimize=False)
    rsi_oversold = IntParam(30, default=30, space='buy', optimize=False)
    bb_period = IntParam(20, default=20, space='buy', optimize=False)
    bb_dev = DecimalParam(2.0, default=2.0, space='buy', optimize=False)
    macd_fast = IntParam(12, default=12, space='buy', optimize=False)
    macd_slow = IntParam(26, default=26, space='buy', optimize=False)
    macd_signal = IntParam(9, default=9, space='buy', optimize=False)
    # DCA parameters
    base_dca_drop_percent = DecimalParam(5.0, default=5.0, space='buy', optimize=False)
    max_n_entry_orders = IntParam(2, default=2, space='buy', optimize=False)  # Max additional entries
    # Exit parameters
    minimal_roi = {
        "0": 0.05,  # 5% profit target
    }
    stoploss = -0.1  # 10% stop-loss
    trailing_stop = True
    trailing_stop_positive = 0.02  # Trail by 2% once profit is reached
    trailing_stop_positive_offset = 0.05  # Activate trailing stop at 5% profit
    trailing_only_offset_is_reached = True
    # Advanced parameters
    atr_period = IntParam(14, default=14, space='buy', optimize=False)
    adx_period = IntParam(14, default=14, space='buy', optimize=False)
    adx_threshold = IntParam(25, default=25, space='buy', optimize=False)
    max_candles_in_trade = IntParam(50, default=50, space='sell', optimize=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate technical indicators for the strategy."""
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=self.bb_period.value, nbdevup=self.bb_dev.value, nbdevdn=self.bb_dev.value)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        # EMA for trend detection
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        # MACD
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast.value, slowperiod=self.macd_slow.value, signalperiod=self.macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define conditions for initial entry with enhanced signals."""
        dataframe.loc[
            (
                (dataframe['rsi'] < self.rsi_oversold.value) &  # RSI indicates oversold
                (dataframe['close'] < dataframe['bb_lower']) &  # Price below lower Bollinger Band
                (dataframe['ema50'] > dataframe['ema200']) &  # Bullish trend confirmed
                (dataframe['macd'] > dataframe['macdsignal'])  # MACD bullish crossover
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define conditions for exiting the trade, including time-based exit."""
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upper']) |  # Price above upper Bollinger Band
                (dataframe['current_time'] - dataframe['entry_time'] > self.max_candles_in_trade.value)  # Time-based exit
            ),
            'exit_long'] = 1
        return dataframe

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, **kwargs):
        """
        Adjust the trade position using a limited DCA approach with dynamic parameters.
        Adds to the position if price drops by a dynamically adjusted percentage and ADX confirms trend strength.
        """
        if trade.is_open and trade.nr_of_successful_entries < self.max_n_entry_orders.value + 1:  # +1 for initial entry
            # Calculate dynamic DCA drop percent based on ATR
            atr_percent = (trade.dataframe['atr'].iloc[-1] / trade.dataframe['close'].iloc[-1]) * 100
            dynamic_dca_drop_percent = self.base_dca_drop_percent.value + atr_percent
            # Calculate price drop since last entry
            last_entry_price = trade.open_rate if trade.nr_of_successful_entries == 1 else trade.open_rate_adjusted
            price_drop = (last_entry_price - current_rate) / last_entry_price * 100
            # Check if ADX confirms strong trend
            adx_value = trade.dataframe['adx'].iloc[-1]
            if price_drop >= dynamic_dca_drop_percent and adx_value > self.adx_threshold.value:
                # Scale stake size based on RSI extremity
                rsi_value = trade.dataframe['rsi'].iloc[-1]
                if rsi_value < 20:  # Extreme oversold
                    stake_amount = trade.stake_amount * 1.5  # Increase stake by 50%
                else:
                    stake_amount = trade.stake_amount / trade.nr_of_successful_entries
                return stake_amount
        return None

    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """Define the initial stake amount."""
        return proposed_stake

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """Set leverage (no leverage used in this strategy)."""
        return 1.0