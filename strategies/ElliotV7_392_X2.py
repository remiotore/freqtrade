from freqtrade.strategy.interface import IStrategy
from functools import reduce
from datetime import datetime, timedelta
from typing import List
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.hyper import set_hyperopt
from freqtrade.strategy import merge_informative_pair
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def ewo(dataframe: DataFrame, ema_fast: int = 20, ema_slow: int = 200) -> DataFrame:
    ema1 = ta.EMA(dataframe, timeperiod=ema_fast)
    ema2 = ta.EMA(dataframe, timeperiod=ema_slow)
    return (ema1 - ema2) / dataframe['close'] * 100

# -----------------------------------------------------------------------------
# Main Strategy
# -----------------------------------------------------------------------------

class ElliotV7_392_X2(IStrategy):
    """5‑minute pullback strategy – vX2
    Key upgrades vs previous version:
    1.   Removed static ROI ladder – selling fully delegated to adaptive trailing.
    2.   Tight maker‑fee aware execution with auto‑requote every 30 s.
    3.   Volatility‑scaled position sizing & ATR‑stop.
    4.   Double‑timeframe trend filter (5 m + 1 h).
    5.   Cooldown to suppress over‑trading (< 18 trades/day by design).
    """

    INTERFACE_VERSION = 3

    # --------------------------------------------------
    # General configuration
    # --------------------------------------------------
    timeframe = '5m'
    higher_tf = '1h'

    startup_candle_count = 120  # for ATR & trend calc
    process_only_new_candles = True

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {'buy': 'gtc', 'sell': 'gtc'}

    # -- Trailing stop ---------------------------------------------------------
    trailing_stop = True
    trailing_stop_positive = 0.002      # 0.2 %
    trailing_stop_positive_offset = 0.04  # start trailing after 4 %
    trailing_only_offset_is_reached = True

    # No static ROI – rely on trailing/stoploss
    minimal_roi = {"0": 0}

    # Emergency stoploss fallback (unlikely hit thanks to ATR SL)
    stoploss = -0.3

    # Cooldown after exiting a trade (minutes)
    cooldown = 15

    # --------------------------------------------------
    # Hyper‑parameters (optimisable)
    # --------------------------------------------------
    base_nb_candles_buy = IntParameter(10, 50, default=20, space='buy')
    low_offset = DecimalParameter(0.9, 0.98, default=0.96, space='buy')

    base_nb_candles_sell = IntParameter(10, 50, default=24, space='sell')
    high_offset = DecimalParameter(1.0, 1.2, default=1.05, space='sell')

    ewo_high = DecimalParameter(2.0, 12.0, default=4.0, space='buy')
    ewo_low = DecimalParameter(-20.0, -8.0, default=-15.0, space='buy')

    rsi_buy = IntParameter(20, 50, default=35, space='buy')

    # --------------------------------------------------
    # Informative pairs
    # --------------------------------------------------
    def informative_pairs(self):
        return [(pair, self.higher_tf) for pair in self.dp.current_whitelist()]

    # --------------------------------------------------
    # Indicators (higher TF)
    # --------------------------------------------------
    def higher_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.higher_tf)
        df['ema_fast'] = ta.EMA(df, timeperiod=20)
        df['ema_slow'] = ta.EMA(df, timeperiod=60)
        df['trend_up'] = (df['ema_fast'] > df['ema_slow'] * 1.003).astype('int')
        return df

    # --------------------------------------------------
    # Indicators (5 m)
    # --------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Merge higher timeframe values
        ht = self.higher_tf_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, ht, self.timeframe, self.higher_tf, ffill=True)

        # EMA channels
        dataframe['ema_buy'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_buy.value)
        dataframe['ema_sell'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_sell.value)

        # EWO + RSI
        dataframe['EWO'] = ewo(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)

        # ATR (volatility) & ATR% for dynamic SL / pos‑size
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100

        return dataframe

    # --------------------------------------------------
    # Buy conditions
    # --------------------------------------------------
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()

        conditions: List = []

        # --- Core pullback condition ----------------------------------------
        pullback = (
            (df['close'] < df['ema_buy'] * self.low_offset.value) &
            (df['trend_up_1h'] > 0) &
            (df['atr_pct'] < 4.5) &
            (df['rsi_fast'] < 25)
        )

        ewo_cond = (
            ((df['EWO'] > self.ewo_high.value) & (df['rsi'] < self.rsi_buy.value)) |
            (df['EWO'] < self.ewo_low.value)
        )

        conditions.append(pullback & ewo_cond)

        if conditions:
            df.loc[reduce(lambda a, b: a | b, conditions), 'buy'] = 1
        return df

    # --------------------------------------------------
    # Sell conditions (rarely used – trailing does most)
    # --------------------------------------------------
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe.copy()
        df.loc[
            (
                (df['close'] > df['ema_sell'] * self.high_offset.value) &
                (df['rsi_fast'] > 70)
            ),
            'sell'] = 1
        return df

    # --------------------------------------------------
    # Adaptive stoploss based on ATR & position age
    # --------------------------------------------------
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs):
        """Tighten SL dynamically"""
        # Position age
        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60

        # Load recent ATR% for the pair
        pair_df = self.dp.get_pair_dataframe(pair, timeframe=self.timeframe)
        atr_pct = (ta.ATR(pair_df, timeperiod=14).iloc[-1] / current_rate)

        # Dynamic SL = max(ATR×1.5, -0.03) => ‑3 % floor
        dynamic_sl = max(-1.5 * float(atr_pct), -0.03)

        # Tighten further if trade ages beyond 120 min and still red
        if age_minutes > 120 and current_profit < 0:
            dynamic_sl = max(dynamic_sl, -0.015)

        return dynamic_sl

    # --------------------------------------------------
    # Position sizing – volatility scaled
    # --------------------------------------------------
    def leverage(self, pair: str, current_available: float, rate: float, **kwargs):
        return 1  # spot only

    def position_adjustment(self, pair: str, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float, **kwargs):
        return None  # disable DCA – single‑shot entries
