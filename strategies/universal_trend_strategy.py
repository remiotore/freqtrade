from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta


class UniversalTrendStrategy(IStrategy):
    """
    Universal Trend Strategy for Freqtrade:
    - Detects bull, bear, and sideways markets using EMA and ADX
    - Buys on RSI oversold in bull or price dip in sideways (lower Bollinger Band)
    - Sells on RSI overbought in bull or price peak in sideways (upper Bollinger Band)
    - Hyperopt-friendly parameters
    """

    # Minimal ROI table: adjust via hyperopt if desired
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0
    }

    # Optimal stoploss (10%)
    stoploss = -0.10

    # Trailing stop settings
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # Use exit signal instead of default ROI/stoploss
    use_exit_signal = True
    process_only_new_candles = True
    timeframe = '5m'

    # Hyperopt parameters
    fast_ema = IntParameter(10, 50, default=20, space='buy')
    slow_ema = IntParameter(100, 200, default=100, space='buy')
    rsi_period = IntParameter(7, 14, default=14, space='buy')
    rsi_buy = IntParameter(20, 40, default=30, space='buy')
    rsi_sell = IntParameter(60, 80, default=70, space='sell')
    adx_period = IntParameter(14, 14, default=14, space='buy')
    adx_trend = IntParameter(20, 40, default=25, space='buy')
    bb_period = IntParameter(20, 30, default=20, space='buy')
    bb_dev = DecimalParameter(1.5, 3.0, default=2.0, space='buy')

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # EMAs for trend detection
        df['ema_fast'] = ta.EMA(df, timeperiod=self.fast_ema.value)
        df['ema_slow'] = ta.EMA(df, timeperiod=self.slow_ema.value)

        # RSI for momentum
        df['rsi'] = ta.RSI(df, timeperiod=self.rsi_period.value)

        # Bollinger Bands for potential range trades
        upper, middle, lower = ta.BBANDS(
            df['close'],
            timeperiod=self.bb_period.value,
            nbdevup=self.bb_dev.value,
            nbdevdn=self.bb_dev.value
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        # ADX for trend strength
        df['adx'] = ta.ADX(df, timeperiod=self.adx_period.value)

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = False

        # Bull trend: fast EMA above slow EMA and ADX indicates strong trend
        bull = (df['ema_fast'] > df['ema_slow']) & (df['adx'] > self.adx_trend.value)
        cond1 = bull & (df['rsi'] < self.rsi_buy.value)

        # Sideways: EMAs converged and low ADX; buy at lower BB
        side = (abs(df['ema_fast'] - df['ema_slow']) < (df['ema_slow'] * 0.001)) & (df['adx'] < self.adx_trend.value)
        cond2 = side & (df['close'] < df['bb_lower'])

        df.loc[cond1 | cond2, 'enter_long'] = True
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = False

        # Exit in bull: RSI overbought
        bull = (df['ema_fast'] > df['ema_slow']) & (df['adx'] > self.adx_trend.value)
        df.loc[bull & (df['rsi'] > self.rsi_sell.value), 'exit_long'] = True

        # Exit in sideways: price above upper BB
        side = (abs(df['ema_fast'] - df['ema_slow']) < (df['ema_slow'] * 0.001)) & (df['adx'] < self.adx_trend.value)
        df.loc[side & (df['close'] > df['bb_upper']), 'exit_long'] = True

        return df
