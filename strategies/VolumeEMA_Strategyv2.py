#v0.1.3
import logging
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import stratninja
pd.set_option('future.no_silent_downcasting', True)

logger = logging.getLogger(__name__)

def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

class VolumeEMA_Strategy(IStrategy):
    INTERFACE_VERSION = 2

    # Main timeframe is 5m
    timeframe = '5m'
    # Disable default ROI and stoploss so that only our logic triggers exits
    stoploss = -0.99
    minimal_roi = {"0": 100}
    startup_candle_count = 4032  # 14 days of 5m candles

    # Volume spike parameters
    volume_lookback = 610
    volume_threshold = 2.5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Volume-based indicators
        dataframe['volume_sma'] = dataframe['volume'].rolling(self.volume_lookback).mean()
        dataframe['volume_std'] = dataframe['volume'].rolling(self.volume_lookback).std()
        dataframe['volume_zscore'] = (dataframe['volume'] - dataframe['volume_sma']) / dataframe['volume_std']
        dataframe['volume_spike'] = dataframe['volume_zscore'] > self.volume_threshold
        dataframe['volume_spike_consecutive'] = dataframe['volume_spike'] & dataframe['volume_spike'].shift(1).fillna(False)

        # ATR calculations
        dataframe['atr_24h'] = ta.ATR(dataframe, timeperiod=288)
        dataframe['avg_atr_14d'] = rma(dataframe['atr_24h'], period=4032)

        # Daily range (24h)
        dataframe['rolling_high_24h'] = dataframe['high'].rolling(288, min_periods=288).max()
        dataframe['rolling_low_24h'] = dataframe['low'].rolling(288, min_periods=288).min()
        dataframe['rolling_dtr_24h'] = dataframe['rolling_high_24h'] - dataframe['rolling_low_24h']

        # EMAs
        for period in [5, 12, 34, 50, 180, 200]:
            dataframe[f'ema{period}'] = ta.EMA(dataframe['close'], timeperiod=period)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        volume_condition = dataframe['volume_spike_consecutive']
        adx_condition = dataframe['adx'] > 25

        # Long entry conditions
        long_ema_condition = dataframe[['ema5', 'ema12']].min(axis=1) > dataframe[['ema34', 'ema50']].max(axis=1)
        long_price_condition = (
            (dataframe['close'] > dataframe['ema5']) &
            (dataframe['close'] > dataframe['ema12']) &
            (dataframe['close'] > dataframe['ema34']) &
            (dataframe['close'] > dataframe['ema50'])
        )
        atr_condition_long = (
            (dataframe['atr_24h'] < 0.9 * dataframe['avg_atr_14d']) |
            (
                (dataframe['atr_24h'] >= 0.9 * dataframe['avg_atr_14d']) &
                (dataframe['close'] < (dataframe['rolling_high_24h'] - 0.3 * dataframe['rolling_dtr_24h']))
            )
        )
        long_entry = volume_condition & adx_condition & long_ema_condition & long_price_condition & atr_condition_long
        dataframe.loc[long_entry, 'enter_long'] = 1
        dataframe.loc[long_entry, 'enter_tag'] = "Long entry: volume, ADX, EMA and ATR conditions met"

        # Short entry conditions
        short_ema_condition = dataframe[['ema5', 'ema12']].max(axis=1) < dataframe[['ema34', 'ema50']].min(axis=1)
        short_price_condition = (
            (dataframe['close'] < dataframe['ema5']) &
            (dataframe['close'] < dataframe['ema12']) &
            (dataframe['close'] < dataframe['ema34']) &
            (dataframe['close'] < dataframe['ema50'])
        )
        atr_condition_short = (
            (dataframe['atr_24h'] < 0.9 * dataframe['avg_atr_14d']) |
            (
                (dataframe['atr_24h'] >= 0.9 * dataframe['avg_atr_14d']) &
                (dataframe['close'] > (dataframe['rolling_low_24h'] + 0.3 * dataframe['rolling_dtr_24h']))
            )
        )
        short_entry = volume_condition & adx_condition & short_ema_condition & short_price_condition & atr_condition_short
        dataframe.loc[short_entry, 'enter_short'] = 1
        dataframe.loc[short_entry, 'enter_tag'] = "Short entry: volume, ADX, EMA and ATR conditions met"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic based solely on stop-loss conditions.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # Long stop-loss only
        long_sl = (dataframe['close'] < dataframe['ema34']) & (dataframe['close'] < dataframe['ema50'])
        dataframe.loc[long_sl, 'exit_long'] = 1

        # Short stop-loss only
        short_sl = (dataframe['close'] > dataframe['ema34']) & (dataframe['close'] > dataframe['ema50'])
        dataframe.loc[short_sl, 'exit_short'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            raise ValueError("No dataframe available in custom_exit.")
        
        required_cols = ['ema34', 'ema50', 'ema5', 'ema12']
        if not all(col in dataframe.columns for col in required_cols):
            dataframe = self.populate_indicators(dataframe, {})
        
        if not all(col in dataframe.columns for col in required_cols):
            raise ValueError("Custom exit: Required indicators missing in the dataframe. "
                             "Ensure that the data has enough candles (startup_candle_count) to compute them.")
        
        last_row = dataframe.iloc[-1]
        reason = None

        if not trade.is_short:  # Long trade
            if last_row['close'] < last_row['ema34'] and last_row['close'] < last_row['ema50']:
                reason = "stop loss: price below EMA34 and EMA50"
            elif last_row['close'] < last_row['ema5'] and last_row['close'] < last_row['ema12'] and current_profit > 0:
                reason = "take profit: price below EMA5 and EMA12"
        else:  # Short trade
            if last_row['close'] > last_row['ema34'] and last_row['close'] > last_row['ema50']:
                reason = "stop loss: price above EMA34 and EMA50"
            elif last_row['close'] > last_row['ema5'] and last_row['close'] > last_row['ema12'] and current_profit > 0:
                reason = "take profit: price above EMA5 and EMA12"

        return reason

    def on_tick(self, pair: str, current_time, current_rate, current_volume, **kwargs):
        dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return

        last_closed = dataframe.iloc[-1]
        current_candle = self.dp.get_current_candle(pair, self.timeframe)
        if current_candle is None:
            return

        volume_sma = last_closed['volume_sma']
        volume_std = last_closed['volume_std']
        if volume_std == 0:
            return

        current_volume_zscore = (current_candle['volume'] - volume_sma) / volume_std

        # Early long entry
        if current_volume_zscore > self.volume_threshold and last_closed['volume_spike']:
            if (current_rate > last_closed['ema5'] and current_rate > last_closed['ema12'] and
                current_rate > last_closed['ema34'] and current_rate > last_closed['ema50']):
                if last_closed['adx'] > 25:
                    if ((last_closed['atr_24h'] < 0.9 * last_closed['avg_atr_14d']) or
                       ((last_closed['atr_24h'] >= 0.9 * last_closed['avg_atr_14d']) and
                        current_rate < (last_closed['rolling_high_24h'] - 0.3 * last_closed['rolling_dtr_24h']))):
                        self.buy(pair, current_rate, tag="Early long entry: volume, ADX, EMA and ATR conditions met")

        # Early short entry
        if current_volume_zscore > self.volume_threshold and last_closed['volume_spike']:
            if (current_rate < last_closed['ema5'] and current_rate < last_closed['ema12'] and
                current_rate < last_closed['ema34'] and current_rate < last_closed['ema50']):
                if last_closed['adx'] > 25:
                    if ((last_closed['atr_24h'] < 0.9 * last_closed['avg_atr_14d']) or
                       ((last_closed['atr_24h'] >= 0.9 * last_closed['avg_atr_14d']) and
                        current_rate > (last_closed['rolling_low_24h'] + 0.3 * last_closed['rolling_dtr_24h']))):
                        self.sell(pair, current_rate, tag="Early short entry: volume, ADX, EMA and ATR conditions met")
