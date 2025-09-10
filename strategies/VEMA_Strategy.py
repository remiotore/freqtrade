import pandas as pd
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass

import logging
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import stratninja

logger = logging.getLogger(__name__)

def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

class VolumeEMA_Strategy(IStrategy):
    INTERFACE_VERSION = 2

    # Base timeframe is set to 5m (supported by the exchange)
    timeframe = '5m'
    stoploss = -0.99
    minimal_roi = {"0": 100}
    # 7 days of 10m candles: 7 * 144 = 1008 candles lookback for indicators
    startup_candle_count = 1008

    # Volume parameters (for DCA only, not for entry)
    volume_lookback = 144   # 24h lookback based on 10m candles
    volume_threshold = 2.5  # used for positive trades in DCA

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # If historical data is in 5m candles, aggregate it into 10m candles for indicator calculations.
        if not dataframe.empty:
            if not isinstance(dataframe.index, pd.DatetimeIndex):
                try:
                    dataframe.index = pd.to_datetime(dataframe.index)
                except Exception as e:
                    logger.error(f"Index conversion error: {e}")
            if len(dataframe.index) > 1:
                dt0 = dataframe.index[0]
                dt1 = dataframe.index[1]
                if (dt1 - dt0).total_seconds() == 300:  # 300 seconds = 5 minutes
                    dataframe = dataframe.resample('10T').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

        # Now use the aggregated data for indicator calculations.
        # Volume-based indicators
        dataframe['volume_sma'] = dataframe['volume'].rolling(self.volume_lookback).mean()
        dataframe['volume_std'] = dataframe['volume'].rolling(self.volume_lookback).std()
        dataframe['volume_zscore'] = (dataframe['volume'] - dataframe['volume_sma']) / dataframe['volume_std']
        dataframe['volume_spike'] = dataframe['volume_zscore'] > self.volume_threshold
        dataframe['volume_spike_consecutive'] = dataframe['volume_spike']

        # ATR calculations: use 144 candles (one day of 10m candles)
        dataframe['atr_24h'] = ta.ATR(dataframe, timeperiod=144)
        # Average ATR over 7 days (7 * 144 = 1008 candles)
        dataframe['avg_atr_7d'] = rma(dataframe['atr_24h'], period=1008)

        # Daily range (24h) calculated on 144-candle windows
        dataframe['rolling_high_24h'] = dataframe['high'].rolling(144, min_periods=144).max()
        dataframe['rolling_low_24h'] = dataframe['low'].rolling(144, min_periods=144).min()
        dataframe['rolling_dtr_24h'] = dataframe['rolling_high_24h'] - dataframe['rolling_low_24h']

        # EMAs calculated on the aggregated 10m data
        for period in [5, 12, 34, 50, 180, 200]:
            dataframe[f'ema{period}'] = ta.EMA(dataframe['close'], timeperiod=period)

        # Compute ADX on the same aggregated data (using period 12)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=12)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        adx_condition = dataframe['adx'] > 30

        # Long entry conditions
        long_ema_condition = dataframe[['ema5', 'ema12']].min(axis=1) > dataframe[['ema34', 'ema50']].max(axis=1)
        long_price_condition = (
            (dataframe['close'] > dataframe['ema5']) &
            (dataframe['close'] > dataframe['ema12']) &
            (dataframe['close'] > dataframe['ema34']) &
            (dataframe['close'] > dataframe['ema50'])
        )
        atr_condition_long = (
            (dataframe['atr_24h'] < 0.9 * dataframe['avg_atr_7d']) &
            (dataframe['close'] < (dataframe['rolling_high_24h'] - 0.3 * dataframe['rolling_dtr_24h']))
        )
        long_entry = adx_condition & long_ema_condition & long_price_condition & atr_condition_long
        dataframe.loc[long_entry, 'enter_long'] = 1
        dataframe.loc[long_entry, 'enter_tag'] = "Long entry: ADX, EMA & ATR conditions met"

        # Short entry conditions
        short_ema_condition = dataframe[['ema5', 'ema12']].max(axis=1) < dataframe[['ema34', 'ema50']].min(axis=1)
        short_price_condition = (
            (dataframe['close'] < dataframe['ema5']) &
            (dataframe['close'] < dataframe['ema12']) &
            (dataframe['close'] < dataframe['ema34']) &
            (dataframe['close'] < dataframe['ema50'])
        )
        atr_condition_short = (
            (dataframe['atr_24h'] < 0.9 * dataframe['avg_atr_7d']) &
            (dataframe['close'] > (dataframe['rolling_low_24h'] + 0.3 * dataframe['rolling_dtr_24h']))
        )
        short_entry = adx_condition & short_ema_condition & short_price_condition & atr_condition_short
        dataframe.loc[short_entry, 'enter_short'] = 1
        dataframe.loc[short_entry, 'enter_tag'] = "Short entry: ADX, EMA & ATR conditions met"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        long_sl = (dataframe['close'] < dataframe['ema34']) & (dataframe['close'] < dataframe['ema50'])
        dataframe.loc[long_sl, 'exit_long'] = 1

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
            raise ValueError("Custom exit: Required indicators missing. Ensure sufficient candles (startup_candle_count).")
        
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

        # Early Entry Logic (Volume not required for entry)
        if (current_rate > last_closed['ema5'] and current_rate > last_closed['ema12'] and
            current_rate > last_closed['ema34'] and current_rate > last_closed['ema50']):
            if last_closed['adx'] > 30:
                if current_rate < (last_closed['rolling_high_24h'] - 0.3 * last_closed['rolling_dtr_24h']):
                    self.buy(pair, current_rate, tag="Early long entry: ADX, EMA & ATR conditions met")

        if (current_rate < last_closed['ema5'] and current_rate < last_closed['ema12'] and
            current_rate < last_closed['ema34'] and current_rate < last_closed['ema50']):
            if last_closed['adx'] > 30:
                if current_rate > (last_closed['rolling_low_24h'] + 0.3 * last_closed['rolling_dtr_24h']):
                    self.sell(pair, current_rate, tag="Early short entry: ADX, EMA & ATR conditions met")

        # DCA Logic (includes ADX and ATR conditions)
        try:
            fast_lower = min(last_closed['ema5'], last_closed['ema12'])
            fast_upper = max(last_closed['ema5'], last_closed['ema12'])
            slow_lower = min(last_closed['ema34'], last_closed['ema50'])
            slow_upper = max(last_closed['ema34'], last_closed['ema50'])

            open_trades = self.trade_manager.get_open_trades()
            for trade in open_trades:
                if trade.pair != pair:
                    continue

                if not (last_closed['adx'] > 30):
                    continue

                if not trade.is_short:
                    if not (last_closed['atr_24h'] < 0.9 * last_closed['avg_atr_7d'] and
                            current_rate < (last_closed['rolling_high_24h'] - 0.3 * last_closed['rolling_dtr_24h'])):
                        continue
                else:
                    if not (last_closed['atr_24h'] < 0.9 * last_closed['avg_atr_7d'] and
                            current_rate > (last_closed['rolling_low_24h'] + 0.3 * last_closed['rolling_dtr_24h'])):
                        continue

                if not hasattr(trade, 'custom_info') or trade.custom_info is None:
                    trade.custom_info = {}
                dca_count = trade.custom_info.get('dca_orders', 0)
                if dca_count >= 3:
                    continue

                chosen_cloud = None
                if fast_lower <= last_closed['close'] <= fast_upper:
                    chosen_cloud = (fast_lower, fast_upper)
                elif slow_lower <= last_closed['close'] <= slow_upper:
                    chosen_cloud = (slow_lower, slow_upper)
                if chosen_cloud is None:
                    continue

                if not trade.is_short:
                    if current_rate <= chosen_cloud[1]:
                        continue
                    retracement = current_rate - chosen_cloud[1]
                else:
                    if current_rate >= chosen_cloud[0]:
                        continue
                    retracement = chosen_cloud[0] - current_rate

                if 'first_dca_diff' not in trade.custom_info:
                    trade.custom_info['first_dca_diff'] = retracement
                else:
                    if retracement < trade.custom_info['first_dca_diff']:
                        continue

                if trade.profit < 0:
                    vol_threshold = 1.0
                else:
                    vol_threshold = 2.5
                if current_volume_zscore < vol_threshold:
                    continue

                if 'initial_trade_amount' not in trade.custom_info:
                    trade.custom_info['initial_trade_amount'] = trade.amount
                additional_qty = trade.custom_info['initial_trade_amount']
                new_qty = trade.amount + additional_qty
                self.trade_manager.modify_order(trade, new_amount=new_qty)
                trade.custom_info['dca_orders'] = dca_count + 1
        except Exception as e:
            logger.error(f"DCA logic error: {e}")
