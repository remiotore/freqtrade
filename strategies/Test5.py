# --- Do not remove these libs -------------------------------------------------
from typing import Dict

import pandas as pd
import talib.abstract as ta               # noqa: N811

from freqtrade.strategy import (
    IStrategy,
    informative,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
)
# -----------------------------------------------------------------------------


class Test5(IStrategy):
    """
    AR Digital – Failed-Breakdown  (with the mirror "failed-breakup" for shorts)
    ---------------------------------------------------------------------------
    * Works on 5-minute candles, needs a 1-day informative timeframe.
    * Hyper-optimisable parameters are declared with Freqtrade Parameter objects.
    * Exits at daily R1/R2 (long) or S1/S2 (short).  A hard stop-loss is supplied
      through `custom_stoploss()` so hyperopt can tune it.
    """

    # --- Freqtrade required settings ----------------------------------------
    timeframe                = "5m"
    informative_timeframe    = "1h"
    can_short                = True
    startup_candle_count     = 240
    process_only_new_candles = True
    use_buy_signal           = True
    use_exit_signal          = True
    # ------------------------------------------------------------------------

    # ----------------------- Hyper-optimisable params ------------------------
    buy_min_break   = DecimalParameter(0.001, 0.02, default=0.005, space="buy")
    buy_max_break   = DecimalParameter(0.02,  0.06, default=0.03,  space="buy")
    buy_rsi_len     = IntParameter(8, 20,             default=14,   space="buy")
    buy_rsi_max     = IntParameter(45, 60,            default=50,   space="buy")
    buy_use_rsi     = CategoricalParameter([True, False], default=True, space="buy")

    sell_tp1_pct    = DecimalParameter(0.005, 0.04, default=0.015, space="sell")
    sell_tp2_pct    = DecimalParameter(0.03,  0.10, default=0.06,  space="sell")

    hard_stoploss   = DecimalParameter(0.03, 0.12, default=0.06, space="sell")
    # ------------------------------------------------------------------------

    # A dummy ROI table – keep hyperopt simple (explicit exits do the work)
    minimal_roi = {
        "0": 10.0
    }

    # A *maximum* stop-loss. Real SL returned via `custom_stoploss()`.
    stoploss = -0.06

    # -------------- Informative indicators ----------------------------------
    @informative('1d')
    def populate_indicators_1d(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate daily pivot levels that will be used for entries and exits.
        """
        # Calculate pivot levels
        dataframe['pivot'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['r1'] = 2 * dataframe['pivot'] - dataframe['low']
        dataframe['r2'] = dataframe['pivot'] + (dataframe['high'] - dataframe['low'])
        dataframe['s1'] = 2 * dataframe['pivot'] - dataframe['high']
        dataframe['s2'] = dataframe['pivot'] - (dataframe['high'] - dataframe['low'])
        
        # Get yesterday's values by shifting
        dataframe['y_high'] = dataframe['high'].shift(1)
        dataframe['y_low'] = dataframe['low'].shift(1)
        dataframe['y_close'] = dataframe['close'].shift(1)
        dataframe['y_pivot'] = dataframe['pivot'].shift(1)
        dataframe['y_r1'] = dataframe['r1'].shift(1)
        dataframe['y_r2'] = dataframe['r2'].shift(1)
        dataframe['y_s1'] = dataframe['s1'].shift(1)
        dataframe['y_s2'] = dataframe['s2'].shift(1)
        
        return dataframe

    # -------------- Indicator calculation -----------------------------------
    def populate_indicators(
        self,
        df: pd.DataFrame,
        metadata: Dict,
    ) -> pd.DataFrame:
        # The informative data is automatically merged by the @informative decorator
        # We can directly use the columns with _1d suffix
        
        # Use yesterday's pivot levels for current trading decisions
        df["pivot"] = df["y_pivot_1d"]
        df["r1"] = df["y_r1_1d"]
        df["r2"] = df["y_r2_1d"]
        df["s1"] = df["y_s1_1d"]
        df["s2"] = df["y_s2_1d"]
        
        # Also make yesterday's OHLC available without suffix for easier access
        df["y_high"] = df["y_high_1d"]
        df["y_low"] = df["y_low_1d"]
        df["y_close"] = df["y_close_1d"]

        # RSI
        df["rsi"] = ta.RSI(df, timeperiod=self.buy_rsi_len.value)

        return df

    # -------------------------- Entry helpers --------------------------------
    def _long_conditions(self, df: pd.DataFrame) -> pd.Series:
        m1, m2 = self.buy_min_break.value, self.buy_max_break.value

        has_support   = df["y_low"].notna()
        broken_down   = (df["y_low"] * (1 - m2) <= df["low"]) & \
                        (df["low"] <= df["y_low"] * (1 - m1))
        bullish_close = df["close"] > df["open"]
        reclaim       = df["close"] > df["y_low"]

        cond = has_support & broken_down & bullish_close & reclaim
        if self.buy_use_rsi.value:
            cond &= df["rsi"] < self.buy_rsi_max.value
        return cond

    def _short_conditions(self, df: pd.DataFrame) -> pd.Series:
        m1, m2 = self.buy_min_break.value, self.buy_max_break.value

        has_resist    = df["y_high"].notna()
        broken_up     = (df["y_high"] * (1 + m1) <= df["high"]) & \
                        (df["high"] <= df["y_high"] * (1 + m2))
        bearish_close = df["close"] < df["open"]
        reclaim       = df["close"] < df["y_high"]

        cond = has_resist & broken_up & bearish_close & reclaim
        if self.buy_use_rsi.value:
            cond &= df["rsi"] > (100 - self.buy_rsi_max.value)
        return cond

    # ------------------------- populate_entry_trend --------------------------
    def populate_entry_trend(
        self,
        df: pd.DataFrame,
        metadata: Dict,
    ) -> pd.DataFrame:
        df.loc[self._long_conditions(df),  ["enter_long",  "enter_tag"]] = (1, "Failed-Breakdown")
        df.loc[self._short_conditions(df), ["enter_short", "enter_tag"]] = (1, "Failed-Breakup")
        return df

    # -------------------------- Exit helpers ---------------------------------
    def _long_exit(self, df: pd.DataFrame) -> pd.Series:
        tp1, tp2 = 1 + self.sell_tp1_pct.value, 1 + self.sell_tp2_pct.value
        return (df["close"] >= df["r1"] * tp1) | (df["close"] >= df["r2"] * tp2)

    def _short_exit(self, df: pd.DataFrame) -> pd.Series:
        tp1, tp2 = 1 - self.sell_tp1_pct.value, 1 - self.sell_tp2_pct.value
        return (df["close"] <= df["s1"] * tp1) | (df["close"] <= df["s2"] * tp2)

    # ------------------------ populate_exit_trend ----------------------------
    def populate_exit_trend(
        self,
        df: pd.DataFrame,
        metadata: Dict,
    ) -> pd.DataFrame:
        df.loc[self._long_exit(df),  ["exit_long",  "exit_tag"]] = (1, "Pivot-TP")
        df.loc[self._short_exit(df), ["exit_short", "exit_tag"]] = (1, "Pivot-TP")
        return df

    # -------------------------- Custom stop-loss -----------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate,
        current_profit,
        **kwargs,
    ):
        """
        Hard SL optimised via `hard_stoploss`.  Return negative value (e.g. -0.06
        for -6 %) so that Freqtrade applies it.
        """
        return -self.hard_stoploss.value