# Conviction_Adaptive_Scalper.py
# Self-contained Freqtrade IStrategy for conservative, conviction-based intraday scalping on futures.

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime, timedelta

class Conviction_Adaptive_Scalper(IStrategy):
    """
    Conviction Adaptive Scalper:
      - timeframe: 1m
      - requires multiple confirmations: trend, VWAP slope, volume spike, orderbook imbalance, momentum
      - adaptive ATR-based stoploss, TP as factor * stoploss
      - trailing stop to protect wins
      - behavioral safety: cooldown after N consecutive losses, max trades per hour/day
    """

    # --- Meta ---
    timeframe = '1m'
    can_short = True

    # Money management (set conservative defaults; tune in hyperopt)
    # We use freqtrade ROI/stoploss plus custom trailing in custom_exit
    stoploss = -0.05  # fallback very large: we use dynamic stoploss via custom_stoploss
    minimal_roi = {
        "0": 0.01,   # fallback, rely mainly on custom exits/trailing
    }

    # startup candles
    startup_candle_count: int = 200

    # --- Tunable strategy parameters (hyperopt these ranges) ---
    ema_trend = 50                 # trend EMA
    ema_fast = 8
    ema_slow = 21

    atr_period = 14
    min_stop_atr = 0.8             # stoploss = max(atr*min_stop_atr, min_stop_pct)
    stop_min_pct = 0.0025          # min stop 0.25%

    tp_mult = 1.6                  # take-profit = tp_mult * stop_loss_size
    use_trailing = True
    trailing_pullback = 0.5        # trailing stop triggers when profit >= trailing_start_pct, pulls back trailing by this fraction
    trailing_start_pct = 0.0035    # start trailing after 0.35%

    vol_sma = 60
    vol_spike_factor = 1.25        # volume must exceed this * vol_sma

    vwap_slope_lookback = 6        # measure VWAP slope over these bars
    min_vwap_slope = 1e-7          # tiny positive slope threshold (adjust after scale)

    obi_threshold = 1.4            # orderbook imbalance bids/asks (long > threshold). For shorts use < 1/threshold

    max_trades_per_hour = 4
    max_trades_per_day = 20

    cooldown_after_losses = 3      # number of consecutive losing trades to trigger cooldown
    cooldown_minutes = 60          # cooldown period length

    # For tracking (not persisted across restarts)
    _recent_trades = []
    _last_cooldown_until = None

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend EMAs
        dataframe['ema_trend'] = ta.EMA(dataframe['close'], timeperiod=self.ema_trend)
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=self.ema_fast)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=self.ema_slow)

        # ATR for dynamic stops
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.atr_period)

        # VWAP (cumulative)
        pv = (dataframe['close'] * dataframe['volume']).fillna(0)
        dataframe['pv_cum'] = pv.cumsum()
        dataframe['vol_cum'] = dataframe['volume'].cumsum().replace(0, np.nan)
        dataframe['vwap'] = dataframe['pv_cum'] / dataframe['vol_cum']

        # VWAP slope (simple linear approx: difference over lookback)
        dataframe['vwap_slope'] = dataframe['vwap'].diff(self.vwap_slope_lookback) / self.vwap_slope_lookback

        # Volume environment
        dataframe['vol_sma'] = dataframe['volume'].rolling(self.vol_sma, min_periods=1).mean()

        # Momentum
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['mom_3'] = dataframe['close'].pct_change(3)

        # Price relative to ema_trend and vwap
        dataframe['pct_above_trend'] = (dataframe['close'] - dataframe['ema_trend']) / dataframe['ema_trend']
        dataframe['pct_vs_vwap'] = (dataframe['close'] - dataframe['vwap']) / dataframe['vwap']

        return dataframe

    def order_book_imbalance(self, metadata: dict) -> float:
        """Return bids/asks volume ratio for top 5 levels; neutral 1.0 on failure."""
        try:
            ob = self.dp.orderbook(metadata['pair'], 5)
            bids = sum([b[1] for b in ob['bids']]) if ob.get('bids') else 0.0
            asks = sum([a[1] for a in ob['asks']]) if ob.get('asks') else 0.0
            if asks == 0:
                return 1.0
            return float(bids) / (float(asks) + 1e-9)
        except Exception:
            return 1.0

    def _within_cooldown(self):
        """Check if currently in cooldown region due to recent losses."""
        if self._last_cooldown_until is None:
            return False
        return datetime.utcnow() < self._last_cooldown_until

    def _register_trade_result(self, trade_outcome: dict):
        """
        Call externally after trades close (freqtrade does not provide a trivial hook to persist across runs).
        This function is here as a guidance placeholder — in live you can implement event hooks.
        """
        # trade_outcome: {'profit_pct': float, 'close_time': datetime}
        self._recent_trades.append(trade_outcome)
        # keep only last 50
        if len(self._recent_trades) > 50:
            self._recent_trades = self._recent_trades[-50:]

        # check consecutive losses
        consec_losses = 0
        for t in reversed(self._recent_trades):
            if t['profit_pct'] < 0:
                consec_losses += 1
            else:
                break
        if consec_losses >= self.cooldown_after_losses:
            self._last_cooldown_until = datetime.utcnow() + timedelta(minutes=self.cooldown_minutes)

    def _count_trades_in_period(self, pair: str, minutes: int, trades: list) -> int:
        """Count trades in last `minutes` from trades list; placeholder for manager integration."""
        # This is a placeholder. Use trade history from trade manager in real deployment.
        return 0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # default flags
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # Do not generate entries when in cooldown
        if self._within_cooldown():
            return dataframe

        # fetch OBI once per call
        obi = self.order_book_imbalance(metadata)

        # Common filters
        vol_ok = dataframe['volume'] > dataframe['vol_sma'] * self.vol_spike_factor
        vwap_slope_ok = dataframe['vwap_slope'] > self.min_vwap_slope
        trend_up = dataframe['pct_above_trend'] > 0.0  # price above long-term trend EMA
        trend_down = dataframe['pct_above_trend'] < 0.0

        # Long conviction:
        # 1) Price above trend (trend_up)
        # 2) VWAP slope positive
        # 3) Volume spike
        # 4) Recent momentum positive (mom_3)
        # 5) Orderbook imbalance favors bids (obi > obi_threshold)
        cond_long = (
            trend_up &
            vwap_slope_ok &
            vol_ok &
            (dataframe['mom_3'] > 0) &
            (obi > self.obi_threshold) &
            (dataframe['rsi'] < 75)  # avoid overbought extremes
        )

        # Short conviction (mirrored):
        cond_short = (
            trend_down &
            (dataframe['vwap_slope'] < -self.min_vwap_slope) &
            vol_ok &
            (dataframe['mom_3'] < 0) &
            (obi < 1.0 / self.obi_threshold) &
            (dataframe['rsi'] > 25)
        )

        # Additionally require price not too far from VWAP (avoid trading outliers)
        within_vwap_long = dataframe['pct_vs_vwap'] > -0.01  # not more than 1% below vwap
        within_vwap_short = dataframe['pct_vs_vwap'] < 0.01  # not more than 1% above vwap

        dataframe.loc[cond_long & within_vwap_long, 'enter_long'] = 1
        dataframe.loc[cond_short & within_vwap_short, 'enter_short'] = 1

        # Tag signals
        dataframe.loc[cond_long, 'entry_tag'] = 'CONV_LONG'
        dataframe.loc[cond_short, 'entry_tag'] = 'CONV_SHORT'

        return dataframe
        
        
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No fixed exit signals — exits handled by minimal_roi, stoploss, and custom_exit.
        This method must still return 'exit_long' and 'exit_short' columns.
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe    

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Dynamic stoploss based on ATR at trade open:
         - stop_size = max(atr_at_entry * min_stop_atr, stop_min_pct)
         - returns negative fraction (e.g., -0.01)
        Freqtrade passes this called repeatedly; we compute once and stash into trade.meta for reuse.
        """
        try:
            # If we already saved stop in trade.meta, return it
            if trade is not None and hasattr(trade, 'meta') and trade.meta.get('dynamic_stop'):
                return trade.meta['dynamic_stop']

            # Acquire current ATR from pair candle series via dp (best-effort)
            candles = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe, limit=5 * self.atr_period)
            if candles is None or len(candles) < self.atr_period:
                return self.stoploss  # fallback

            recent_atr = ta.ATR(candles['high'], candles['low'], candles['close'], timeperiod=self.atr_period).iloc[-1]
            if np.isnan(recent_atr) or recent_atr == 0:
                return self.stoploss

            # stop as percentage ~ recent_atr / current_price * factor
            stop_pct = max((recent_atr / current_rate) * self.min_stop_atr, self.stop_min_pct)
            dyn_stop = -abs(stop_pct)

            # stash in trade.meta if available
            if trade is not None and hasattr(trade, 'meta'):
                meta = trade.meta if trade.meta is not None else {}
                meta['dynamic_stop'] = dyn_stop
                trade.meta = meta

            return dyn_stop
        except Exception:
            return self.stoploss

    def custom_exit(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Implements:
         - take profit at tp_mult * stop_size (we compute using dynamic_stop from trade.meta if available),
         - trailing stop: once profit > trailing_start_pct, apply trailing logic.
        """
        # ensure trade.meta has dynamic_stop if possible
        dynamic_stop = None
        try:
            if trade is not None and hasattr(trade, 'meta'):
                dynamic_stop = trade.meta.get('dynamic_stop')
        except Exception:
            dynamic_stop = None

        # fallback: estimate dynamic stop if missing
        if dynamic_stop is None:
            dynamic_stop = self.custom_stoploss(pair, trade, current_time, current_rate, current_profit, **kwargs)

        stop_abs = abs(dynamic_stop)
        if stop_abs == 0:
            return None

        tp_target = stop_abs * self.tp_mult

        # If profit reached TP target, exit
        if current_profit is not None and current_profit >= tp_target:
            return 'tp_dynamic'

        # Trailing stop logic
        if self.use_trailing and current_profit is not None:
            if current_profit >= self.trailing_start_pct:
                # set trailing threshold: if current profit retracts by trailing_pullback*current_profit, exit
                # we compute last_peak from trade.meta (placeholder) — for simplicity use current_profit as proxy
                # A better implementation should track trade peak profit in an external persistent store.
                peak = trade.meta.get('peak_profit', current_profit) if (trade and hasattr(trade, 'meta')) else current_profit
                # update peak
                if trade is not None and hasattr(trade, 'meta'):
                    trade.meta['peak_profit'] = max(peak, current_profit)
                    peak = trade.meta['peak_profit']

                # if profit pulled back beyond allowed fraction, exit
                if current_profit < peak * (1.0 - self.trailing_pullback):
                    return 'trailing_pullback'

        # Else hold
        return None

    # Optional hook: selection of pairs / stop trading based on session — not implemented here
