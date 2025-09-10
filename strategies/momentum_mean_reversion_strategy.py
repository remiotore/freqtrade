from datetime import datetime, timedelta
import warnings

import talib.abstract as ta
import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, DecimalParameter
from functools import reduce

# Suppress runtime warnings from indicator calculations\ nwarnings.simplefilter(action="ignore", category=RuntimeWarning)

# Buffers to track trade IDs for custom exit logic
_pending_hold = []
_pending_hold_alt = []


class MomentumMeanReversionV1(IStrategy):
    """
    A momentum + mean-reversion trading strategy for futures with\ ncustom entry & exit rules, dynamic ROI targets, and trailing stops.
    """

    # -- Return targets: minute: ROI
    minimal_roi = {
        "0":    0.12,   # immediate 12%
        "30":   0.05,   #  5% after 30 minutes
        "60":   0.02,   #  2% after 1 hour
        "120":  0.01,   #  1% after 2 hours
        "240":  0.005,  # 0.5% after 4 hours
        "480":  0.0     # break-even at 8 hours
    }

    # Strategy settings
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 240
    leverage_level = 5

    # Hard stoploss and trailing-stop parameters
    stoploss = -0.28
    trailing_stop = True
    trailing_stop_positive = 0.009
    trailing_stop_positive_offset = 0.026
    trailing_only_offset_is_reached = True

    # -- Optimization flags and hyperparameters
    _opt_enabled = True
    buy_rsi_fast = IntParameter(20, 70, default=40, space='buy', optimize=_opt_enabled)
    buy_rsi_main = IntParameter(15, 50, default=42, space='buy', optimize=_opt_enabled)
    buy_sma_ratio = DecimalParameter(0.900, 1.0, default=0.973, decimals=3, space='buy', optimize=_opt_enabled)
    buy_cti_thresh = DecimalParameter(-1.0, 1.0, default=0.69, decimals=2, space='buy', optimize=_opt_enabled)

    sell_fastk_thresh = IntParameter(50, 100, default=84, space='sell', optimize=True)

    # CCI-based exit loss prevention
    _cci_opt = True
    sell_cci_loss_level = IntParameter(0, 600, default=120, space='sell', optimize=_cci_opt)
    sell_cci_loss_profit = DecimalParameter(-0.15, 0.0, default=-0.15, decimals=2, space='sell', optimize=_cci_opt)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate and store all necessary indicators in the DataFrame.
        """
        # Simple moving averages
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['ma120'] = ta.MA(dataframe, timeperiod=120)
        dataframe['ma240'] = ta.MA(dataframe, timeperiod=240)

        # Relative Strength Index: fast, main, slow
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Composite Trend Index for extra momentum detection
        dataframe['cti'] = pta.cti(dataframe['close'], length=20)

        # Stochastic Fast K line
        stoch = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch['fastk']

        # Commodity Channel Index
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Mark entry signals when momentum dips but price aligns below short-term SMA.
        """
        dataframe['enter_tag'] = ''
        conditions = []

        # Primary entry condition
        cond1 = (
            (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift()) &
            (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
            (dataframe['rsi'] > self.buy_rsi_main.value) &
            (dataframe['close'] < dataframe['sma_15'] * self.buy_sma_ratio.value) &
            (dataframe['cti'] < self.buy_cti_thresh.value)
        )
        conditions.append(cond1)
        dataframe.loc[cond1, 'enter_tag'] += 'primary'

        # Alternative entry with fixed thresholds
        cond2 = (
            (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift()) &
            (dataframe['rsi_fast'] < 34) &
            (dataframe['rsi'] > 28) &
            (dataframe['close'] < dataframe['sma_15'] * 0.96) &
            (dataframe['cti'] < self.buy_cti_thresh.value)
        )
        conditions.append(cond2)
        dataframe.loc[cond2, 'enter_tag'] += 'fallback'

        # Combine all conditions into enter_long flag
        if conditions:
            combined = reduce(lambda a, b: a | b, conditions)
            dataframe.loc[combined, 'enter_long'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> str | None:
        """
        Override exit signals based on profit and indicator crosses.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        row = df.iloc[-1]

        # Track in/out of MA zones for potential hold flags
        if row['close'] > row['ma120'] or row['close'] > row['ma240']:
            if trade.id not in _pending_hold:
                _pending_hold.append(trade.id)
        else:
            if trade.id not in _pending_hold_alt:
                _pending_hold_alt.append(trade.id)

        # Fast-K profit-taking
        if current_profit > 0 and row['fastk'] > self.sell_fastk_thresh.value:
            return 'fastk_profit_sell'

        # CCI breakout exits
        if row['cci'] > 80 and row['high'] >= trade.open_rate:
            return 'cci_high_sell'

        # Loss exit when deeply negative but CCI spiking
        if current_profit <= -0.15 and row['cci'] > 200:
            return 'cci_loss_sell'

        # Default: no custom exit
        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder for exit trend; custom_exit handles all logic.
        """
        dataframe[['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Always use a fixed leverage level.
        """
        return self.leverage_level
