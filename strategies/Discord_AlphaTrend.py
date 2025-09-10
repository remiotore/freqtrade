import logging

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    stoploss_from_open,
)
from datetime import datetime

logger = logging.getLogger(__name__)


class AlphaTrend(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    minimal_roi = {"0": 100.0}

    stoploss = -0.99
    trailing_stop = False
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    startup_candle_count = 14

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    use_custom_stoploss = True
    can_short = True

    # default params
    buy_params = {"alpha_coeff": 1}
    sell_params = {
        "leverage_num": 1,
        # custom stop loss params
        "pHSL": -0.675,
        "pPF_1": 0.069,
        "pPF_2": 0.044,
        "pSL_1": 0.1,
        "pSL_2": 0.149,
    }

    alpha_coeff = DecimalParameter(
        low=0.5,
        high=1.5,
        default=buy_params["alpha_coeff"],
        space="buy",
        optimize=True,
        decimals=1,
    )
    # leverage_num = IntParameter(1, 2, default=1, space="sell")

    # trailing stoploss
    trailing_optimize = True
    pHSL = DecimalParameter(
        -0.990,
        -0.040,
        default=-0.08,
        decimals=3,
        space="sell",
        optimize=trailing_optimize,
    )
    pPF_1 = DecimalParameter(
        0.008,
        0.100,
        default=0.016,
        decimals=3,
        space="sell",
        optimize=trailing_optimize,
    )
    pSL_1 = DecimalParameter(
        0.01,
        0.05,
        default=0.02,
        decimals=2,
        space="sell",
        optimize=trailing_optimize,
    )
    pPF_2 = DecimalParameter(
        0.04,
        0.20,
        default=0.08,
        decimals=2,
        space="sell",
        optimize=trailing_optimize,
    )
    pSL_2 = DecimalParameter(
        0.04,
        0.20,
        default=0.040,
        decimals=2,
        space="sell",
        optimize=trailing_optimize,
    )

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        df["ema_200"] = ta.EMA(df, timeperiod=200)
        df["rsi_14"] = ta.RSI(df, 14)

        # MFI
        df["mfi_14"] = ta.MFI(df, 14)

        # Alpha Trend (Tradingview)
        df["TR"] = ta.TRANGE(df)
        df["ATR"] = ta.SMA(df["TR"], 14)

        for val in self.alpha_coeff.range:
            df[f"alpha_trend_{val}"] = alpha_trend(
                df, df["ATR"], coeff=val, mfi_threshold=50
            )

        return df

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        alpha_buy = qtpylib.crossed_above(
            df[f"alpha_trend_{self.alpha_coeff.value}"],
            df[f"alpha_trend_{self.alpha_coeff.value}"].shift(2),
        )
        alpha_sell = qtpylib.crossed_below(
            df[f"alpha_trend_{self.alpha_coeff.value}"],
            df[f"alpha_trend_{self.alpha_coeff.value}"].shift(2),
        )

        # Enter Long
        O1 = df.groupby(
            alpha_buy.shift().fillna(False).astype(int).eq(1).cumsum()
        ).cumcount()
        K2 = df.groupby(alpha_sell.astype(int).eq(1).cumsum()).cumcount()

        # Enter Short
        K1 = df.groupby(alpha_buy.astype(int).eq(1).cumsum()).cumcount()
        O2 = df.groupby(
            alpha_sell.shift().fillna(False).astype(int).eq(1).cumsum()
        ).cumcount()

        df.loc[
            (df["close"] < df["ema_200"]) & (alpha_buy & (O1 > K2) & df["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "enter_long")

        df.loc[
            (df["close"] > df["ema_200"]) & (alpha_sell & (O2 > K1) & df["volume"] > 0),
            ["enter_short", "enter_tag"],
        ] = (1, "enter_short")

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        alpha_buy = qtpylib.crossed_above(
            df[f"alpha_trend_{self.alpha_coeff.value}"],
            df[f"alpha_trend_{self.alpha_coeff.value}"].shift(2),
        )
        alpha_sell = qtpylib.crossed_below(
            df[f"alpha_trend_{self.alpha_coeff.value}"],
            df[f"alpha_trend_{self.alpha_coeff.value}"].shift(2),
        )

        ## Exit Long
        K1 = df.groupby(alpha_buy.astype(int).eq(1).cumsum()).cumcount()
        O2 = df.groupby(
            alpha_sell.shift().fillna(False).astype(int).eq(1).cumsum()
        ).cumcount()

        ## Exit Short
        O1 = df.groupby(
            alpha_buy.shift().fillna(False).astype(int).eq(1).cumsum()
        ).cumcount()
        K2 = df.groupby(alpha_sell.astype(int).eq(1).cumsum()).cumcount()

        df.loc[
            (df["close"] > df["ema_200"]) & (alpha_sell & (O2 > K1) & df["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "exit_long")

        df.loc[
            (df["close"] < df["ema_200"]) & (alpha_buy & (O1 > K2) & df["volume"] > 0),
            ["exit_short", "exit_tag"],
        ] = (1, "exit_short")

        return df

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:

        return 2  # self.leverage_num.value

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if self.can_short:
            if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
                return 1
        else:
            if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
                return 1

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)


def alpha_trend(
    df: pd.DataFrame, ATR: pd.Series, coeff: float = 1.0, mfi_threshold: int = 50
):
    upT = df["low"] - ATR * coeff
    downT = df["high"] + ATR * coeff

    new_values = []
    previous_val = np.nan
    for idx in df.index:
        if df.loc[idx, "mfi_14"] >= mfi_threshold:
            if upT[idx] < previous_val:
                x = previous_val
            else:
                x = upT[idx]
        else:
            if downT[idx] > previous_val:
                x = previous_val
            else:
                x = downT[idx]
        new_values.append(x)
        previous_val = x

    return pd.Series(new_values, index=df.index)
