# user_data/strategies/QuickGainLowLoss.py
# Freqtrade ≥2024.5, INTERFACE_VERSION 3
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, informative
from datetime import datetime, timedelta
from typing import Optional

class QuickGainLowLoss(IStrategy):
    """
    Rapid EMA–ADX scalping strategy.
    """

    INTERFACE_VERSION = 3
    timeframe              = "5m"
    process_only_new_candles = True
    startup_candle_count    = 50         # Needs 50 to compute ATR on 1h

    # ----- Risk parameters -----
    minimal_roi = {
        "0":     0.007,   # < 20 min
        "20":    0.005,
        "40":    0.004,
        "60":    0.0035
    }

    stoploss                = -0.018     # 1.8 %

    trailing_stop           = True       # failsafe if custom_stoploss blocked
    trailing_stop_positive  = 0.006
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    # ----- Protection layer -----
    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 5},
        {"method": "StoplossGuard", "lookback_period_candles": 1440,
         "trade_limit": 2, "stop_duration_candles": 30, "only_per_pair": False},
    ]

    # ----- Informative timeframe decorator -----
    @informative(timeframe="1h")
    def informative_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema200"] = ta.EMA(dataframe["close"], timeperiod=200)
        dataframe["atr"]    = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"]    = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    # ----- Main indicator block -----
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe["close"], timeperiod=12)
        dataframe["ema_slow"] = ta.EMA(dataframe["close"], timeperiod=26)
        dataframe["adx"]      = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"]      = ta.RSI(dataframe, timeperiod=14)
        # Merge informative TF
        inf_tf = self.informative_indicators(dataframe.copy(), metadata)
        dataframe = qtpylib.merge_informative_pair(dataframe, inf_tf,
                                                   self.timeframe, "1h",
                                                   ffill=True, prefix="1h")
        return dataframe

    # ----- Entry conditions -----
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        long_cond = (
            (df["ema_fast"] > df["ema_slow"]) &
            (df["adx"] > 25) &
            (df["1h_ema200"] < df["1h_close"]) &   # higher TF confirmation
            (df["1h_adx"] > 20) &
            (df["rsi"] < 70) &                     # avoid over-bought spike
            (qtpylib.crossed_above(df["ema_fast"], df["ema_slow"]))
        )
        df.loc[long_cond, "enter_long"] = 1

        short_cond = (
            (df["ema_fast"] < df["ema_slow"]) &
            (df["adx"] > 25) &
            (df["1h_ema200"] > df["1h_close"]) &
            (df["1h_adx"] > 20) &
            (df["rsi"] > 30) &
            (qtpylib.crossed_below(df["ema_fast"], df["ema_slow"]))
        )
        df.loc[short_cond, "enter_short"] = 1
        return df

    # ----- Exit conditions -----
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long = qtpylib.crossed_below(df["ema_fast"], df["ema_slow"])
        exit_short = qtpylib.crossed_above(df["ema_fast"], df["ema_slow"])

        df.loc[exit_long, "exit_long"]   = 1
        df.loc[exit_short, "exit_short"] = 1
        return df

    # ----- Custom adaptive stop-loss -----
    use_custom_stoploss = True
    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # tighten once profit positive
        if current_profit > 0.012:      # ≥ 1.2 %
            return max(-0.005, current_profit * 0.4 * -1)
        elif current_profit > 0.006:    # 0.6–1.2 %
            return -0.003               # lock 0.3 %
        return -0.018                   # default SL
