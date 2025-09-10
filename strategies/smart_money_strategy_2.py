import logging
from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import CategoricalParameter, IntParameter, informative
from freqtrade.strategy.interface import IStrategy
from mixins import SafetyOrderMixin, TrailingTakeProfitMixin
from pandas import DataFrame

logger = logging.getLogger(__name__)


def detect_pullback(dataframe: DataFrame, periods=30, method="pct_outlier"):
    """
    Pullback & Outlier Detection
    Know when a sudden move and possible reversal is coming

    Method 1: StDev Outlier (z-score)
    Method 2: Percent-Change Outlier (z-score)
    Method 3: Candle Open-Close %-Change

    outlier_threshold - Recommended: 2.0 - 3.0

    df['pullback_flag']: 1 (Outlier Up) / -1 (Outlier Down)
    """
    df = dataframe.copy()
    if method == "stdev_outlier":
        outlier_threshold = 2.0
        df["dif"] = df["close"] - df["close"].shift(1)
        df["dif_squared_sum"] = (df["dif"] ** 2).rolling(window=periods + 1).sum()
        df["std"] = np.sqrt(
            (df["dif_squared_sum"] - df["dif"].shift(0) ** 2) / (periods - 1)
        )
        df["z"] = df["dif"] / df["std"]
        df["pullback_flag"] = np.where(df["z"] >= outlier_threshold, 1, 0)
        df["pullback_flag"] = np.where(
            df["z"] <= -outlier_threshold, -1, df["pullback_flag"]
        )

    if method == "pct_outlier":
        outlier_threshold = 2.0
        df["pb_pct_change"] = df["close"].pct_change()
        df["pb_zscore"] = qtpylib.zscore(df, window=periods, col="pb_pct_change")
        df["pullback_flag"] = np.where(df["pb_zscore"] >= outlier_threshold, 1, 0)
        df["pullback_flag"] = np.where(
            df["pb_zscore"] <= -outlier_threshold, -1, df["pullback_flag"]
        )

    if method == "candle_body":
        pullback_pct = 1.0
        df["change"] = df["close"] - df["open"]
        df["pullback"] = (df["change"] / df["open"]) * 100
        df["pullback_flag"] = np.where(df["pullback"] >= pullback_pct, 1, 0)
        df["pullback_flag"] = np.where(
            df["pullback"] <= -pullback_pct, -1, df["pullback_flag"]
        )

    return df


def smi_trend(
    dataframe: DataFrame, k_length=9, d_length=3, smoothing_type="EMA", smoothing=10
):
    """
    Stochastic Momentum Index (SMI) Trend Indicator

    SMI > 0 and SMI > MA: (2) Bull
    SMI < 0 and SMI > MA: (1) Possible Bullish Reversal

    SMI > 0 and SMI < MA: (-1) Possible Bearish Reversal
    SMI < 0 and SMI < MA: (-2) Bear

    Returns:
        pandas.Series: New feature generated
    """
    df = dataframe.copy()
    ll = df["low"].rolling(window=k_length).min()
    hh = df["high"].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df["close"] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)

    if smoothing_type == "SMA":
        smi_ma = ta.SMA(smi, timeperiod=smoothing)
    elif smoothing_type == "EMA":
        smi_ma = ta.EMA(smi, timeperiod=smoothing)
    elif smoothing_type == "WMA":
        smi_ma = ta.WMA(smi, timeperiod=smoothing)
    elif smoothing_type == "DEMA":
        smi_ma = ta.DEMA(smi, timeperiod=smoothing)
    elif smoothing_type == "TEMA":
        smi_ma = ta.TEMA(smi, timeperiod=smoothing)
    else:
        raise ValueError("Choose an MA Type: 'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA'")

    conditions = [
        (np.greater(smi, 0) & np.greater(smi, smi_ma)),  # (2) Bull
        (np.less(smi, 0) & np.greater(smi, smi_ma)),  # (1) Possible Bullish Reversal
        (np.greater(smi, 0) & np.less(smi, smi_ma)),  # (-1) Possible Bearish Reversal
        (np.less(smi, 0) & np.less(smi, smi_ma)),  # (-2) Bear
    ]

    smi_trend = np.select(conditions, [2, 1, -1, -2])

    return smi, smi_ma, smi_trend


def smart_money_index(dataframe: DataFrame):
    df = dataframe.copy()
    last_candle = df.iloc[-1].squeeze()

    df["morning_close"] = df.loc[(df["date"].dt.hour == 1), "close"]
    df["morning_close"] = df["morning_close"].ffill()

    df["afternoon_open"] = df.loc[(df["date"].dt.hour == 6), "open"]
    df["afternoon_open"] = df["afternoon_open"].ffill()

    try:
        last_smi = last_candle["smart_money_index"]
    except KeyError:
        last_smi = 1

    df["smart_money_index"] = 1
    df["smart_money_index"] = (
        last_smi
        - (df["open"] - df["morning_close"])
        + (df["afternoon_open"] - df["close"])
    )

    return df


class smart_money_strategy_2(IStrategy):
    INTERFACE_VERSION = 3

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    stoploss = -0.99  # value loaded from strategy
    """
    END HYPEROPT
    """

    use_exit_signal: bool = True
    exit_profit_only: bool = False
    exit_profit_offset: float = 0.01
    ignore_roi_if_entry_signal: bool = True

    timeframe: str = "5m"

    process_only_new_candles: bool = True
    startup_candle_count: int = 200

    smart_money_index_fast_ma_buy = IntParameter(10, 100, default=20, space="buy")
    smart_money_index_slow_ma_buy = IntParameter(100, 350, default=150, space="buy")
    smart_money_index_fast_ma_sell = IntParameter(10, 100, default=50, space="sell")
    smart_money_index_slow_ma_sell = IntParameter(100, 300, default=150, space="sell")
    smart_money_exit_trigger = CategoricalParameter(
        ["fast", "slow", "disabled"], default="fast", space="sell"
    )

    entry_guard = CategoricalParameter(["ema", "disabled"], default="ema", space="buy")

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema50": {
                    "color": "#26a269",
                },
                "ema50_1h": {
                    "color": "#a51d2d",
                },
            },
            "subplots": {
                "smart_money_index": {
                    "smart_money_index": {"color": "#26a269", "type": "line"},
                    f"smart_money_index_fast_ma_buy_{self.smart_money_index_fast_ma_buy.value}": {
                        "color": "#a51d2d",
                        "type": "line",
                    },
                },
                "smart_money_index_slow": {
                    "smart_money_index_1h": {"color": "#26a269", "type": "line"},
                    f"smart_money_index_slow_ma_buy_{self.smart_money_index_slow_ma_buy.value}_1h": {
                        "color": "#a51d2d",
                        "type": "line",
                    },
                },
            },
        }

    def populate_smart_money_indicators(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()
        df = smart_money_index(df)

        frames = [df]

        for val in self.smart_money_index_slow_ma_buy.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_slow_ma_buy_{val}": ta.SMA(
                            df["smart_money_index"], val
                        )
                    }
                )
            )

        for val in self.smart_money_index_fast_ma_buy.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_fast_ma_buy_{val}": ta.SMA(
                            df["smart_money_index"], val
                        )
                    }
                )
            )

        for val in self.smart_money_index_slow_ma_sell.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_slow_ma_sell_{val}": ta.SMA(
                            df["smart_money_index"], val
                        )
                    }
                )
            )

        for val in self.smart_money_index_fast_ma_sell.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_fast_ma_sell_{val}": ta.SMA(
                            df["smart_money_index"], val
                        )
                    }
                )
            )

        return pd.concat(frames, axis=1)

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.populate_smart_money_indicators(dataframe)
        dataframe["smi"], dataframe["smi_ma"], dataframe["smi_trend"] = smi_trend(
            dataframe
        )
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.populate_smart_money_indicators(dataframe)
        dataframe["smi"], dataframe["smi_ma"], dataframe["smi_trend"] = smi_trend(
            dataframe
        )
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "enter_tag"] = ""

        smart_money_cross = (
            qtpylib.crossed_above(
                dataframe["smart_money_index"],
                dataframe[
                    f"smart_money_index_fast_ma_buy_{self.smart_money_index_fast_ma_buy.value}"
                ],
            )
            & (dataframe["volume"] > 0)
            & (dataframe["smi_trend_1h"] != 2)
            & (dataframe["close"] < dataframe["ema50_1h"])
        )
        dataframe.loc[smart_money_cross, "enter_tag"] += "+fast_smart_money_cross"
        conditions.append(smart_money_cross)

        inf_smart_money_cross = (
            qtpylib.crossed_above(
                dataframe["smart_money_index_1h"],
                dataframe[
                    f"smart_money_index_slow_ma_buy_{self.smart_money_index_slow_ma_buy.value}_1h"
                ],
            )
            & (dataframe["volume"] > 0)
            & (dataframe["smi_trend_1h"] != 2)
            & (dataframe["close"] < dataframe["ema50_1h"])
        )
        dataframe.loc[inf_smart_money_cross, "enter_tag"] += "+slow_smart_money_cross"
        conditions.append(inf_smart_money_cross)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, "exit_tag"] = ""

        if self.smart_money_exit_trigger.value == "fast":
            smart_money_cross = (
                qtpylib.crossed_below(
                    dataframe["smart_money_index"],
                    dataframe[
                        f"smart_money_index_fast_ma_sell_{self.smart_money_index_fast_ma_sell.value}"
                    ],
                )
                & (dataframe["volume"] > 0)
                & (dataframe["smi_trend_1h"] == 2)
            )
            dataframe.loc[smart_money_cross, "enter_tag"] += "+smart_money_cross"
            conditions.append(smart_money_cross)

        elif self.smart_money_exit_trigger.value == "slow":
            inf_smart_money_cross = (
                qtpylib.crossed_below(
                    dataframe["smart_money_index_1h"],
                    dataframe[
                        f"smart_money_index_slow_ma_sell_{self.smart_money_index_slow_ma_sell.value}_1h"
                    ],
                )
                & (dataframe["volume"] > 0)
                & (dataframe["smi_trend_1h"] == 2)
            )
            dataframe.loc[
                inf_smart_money_cross, "enter_tag"
            ] += "+inf_smart_money_cross"
            conditions.append(inf_smart_money_cross)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "exit_long"] = 1

        return dataframe


class SmartMoneyStrategy_DCA(SafetyOrderMixin, SmartMoneyStrategy):
    pass


class SmartMoneyStrategy_DCA_TTP(
    TrailingTakeProfitMixin, SafetyOrderMixin, SmartMoneyStrategy
):
    pass
