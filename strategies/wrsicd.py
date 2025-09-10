



import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)


from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class wrsicd(IStrategy):
    """
    author@: Bryant Suen
    github@: https://github.com/BryantSuen

    Originally designed by @lazybear: https://www.tradingview.com/script/qt6xLfLi-Impulse-MACD-LazyBear/

    """

    INTERFACE_VERSION = 3

    timeframe = "1h"

    can_short: bool = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    buy_params = {
        "cmf_enabled": False,
        "cmf_long_threshold": 0.147,
        "cmf_short_threshold": 0.806,
        "day_ema_enabled": False,
        "day_ema_period": 7,
        "ha_rsi_enabled": False,
        "ha_rsi_long_threshold": 39,
        "ha_rsi_short_threshold": 39,
        "ha_williams_r_enabled": True,
        "ha_williams_r_long_threshold": -73,
        "ha_williams_r_short_threshold": -42,
        "rsi_enabled": False,
        "rsi_long_threshold": 33,
        "rsi_short_threshold": 16,
        "volume_enabled": False,
        "williams_r_enabled": False,
        "williams_r_long_threshold": -69,
        "williams_r_short_threshold": -85,
        "macd_ma_period": 20,  # value loaded from strategy
        "macd_signal_period": 14,  # value loaded from strategy
        
        "check_macd_position": False,
    }

    macd_ma_period = IntParameter(
        10, 30, default=buy_params["macd_ma_period"], space="buy", optimize=False
    )
    macd_signal_period = IntParameter(
        3, 20, default=buy_params["macd_signal_period"], space="buy", optimize=False
    )
    check_macd_position = BooleanParameter(
        default=buy_params["check_macd_position"], space="sell", optimize=True
    )

    rsi_enabled = BooleanParameter(default=buy_params["rsi_enabled"], space="buy", optimize=True)
    rsi_long_threshold = IntParameter(
        5, 100, default=buy_params["rsi_long_threshold"], space="buy", optimize=True
    )
    rsi_short_threshold = IntParameter(
        5, 100, default=buy_params["rsi_short_threshold"], space="buy", optimize=True
    )

    williams_r_enabled = BooleanParameter(
        default=buy_params["williams_r_enabled"], space="buy", optimize=True
    )
    williams_r_long_threshold = IntParameter(
        -100,
        -5,
        default=buy_params["williams_r_long_threshold"],
        space="buy",
        optimize=True,
    )
    williams_r_short_threshold = IntParameter(
        -100,
        -5,
        default=buy_params["williams_r_short_threshold"],
        space="buy",
        optimize=True,
    )

    ha_rsi_enabled = BooleanParameter(
        default=buy_params["ha_rsi_enabled"], space="buy", optimize=True
    )
    ha_rsi_long_threshold = IntParameter(
        5,
        100,
        default=buy_params["ha_rsi_long_threshold"],
        space="buy",
        optimize=True,
    )
    ha_rsi_short_threshold = IntParameter(
        5,
        100,
        default=buy_params["ha_rsi_short_threshold"],
        space="buy",
        optimize=True,
    )

    ha_williams_r_enabled = BooleanParameter(
        default=buy_params["ha_williams_r_enabled"], space="buy", optimize=True
    )
    ha_williams_r_long_threshold = IntParameter(
        -100,
        -5,
        default=buy_params["ha_williams_r_long_threshold"],
        space="buy",
        optimize=True,
    )
    ha_williams_r_short_threshold = IntParameter(
        -100,
        -5,
        default=buy_params["ha_williams_r_short_threshold"],
        space="buy",
        optimize=True,
    )

    cmf_enabled = BooleanParameter(default=buy_params["cmf_enabled"], space="buy", optimize=True)
    cmf_long_threshold = DecimalParameter(
        -1.0, 1.0, default=buy_params["cmf_long_threshold"], space="buy", optimize=True
    )
    cmf_short_threshold = DecimalParameter(
        -1.0, 1.0, default=buy_params["cmf_short_threshold"], space="buy", optimize=True
    )

    day_ema_enabled = BooleanParameter(
        default=buy_params["day_ema_enabled"], space="buy", optimize=True
    )
    day_ema_period = IntParameter(
        1, 10, default=buy_params["day_ema_period"], space="buy", optimize=True
    )
    volume_enabled = BooleanParameter(
        default=buy_params["volume_enabled"], space="buy", optimize=True
    )

    minimal_roi = {"0": 0.9}

    stoploss = -0.25

    trailing_stop = True
    trailing_stop_positive = 0.07

    trailing_stop_positive_offset = 0.22
    trailing_only_offset_is_reached = True

    process_only_new_candles = False
    use_exit_signal = False

    startup_candle_count: int = 100

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1d") for pair in pairs]

        return informative_pairs

    def _cal_smma(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _cal_zero_lag_ema(self, series: pd.Series, period: int) -> pd.Series:
        ema_1 = ta.EMA(series, timeperiod=period)
        ema_2 = ta.EMA(ema_1, timeperiod=period)
        return 2 * ema_1 - ema_2

    def _cal_volume_divergence(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
        volume_positive = dataframe.apply(
            lambda x: x["volume"] if x["close"] > x["open"] else 0, axis=1
        )
        volume_negative = dataframe.apply(
            lambda x: x["volume"] if x["close"] < x["open"] else 0, axis=1
        )
        volume_divergence = ta.EMA(volume_positive - volume_negative, period=14) / ta.EMA(
            volume_positive + volume_negative, period=14
        )
        volume_divergence_signal = ta.EMA(volume_divergence, period=period)

        return volume_divergence, volume_divergence_signal

    def impulsive_macd(self, dataframe: DataFrame, length_ma: int, length_signal: int) -> tuple:
        mean_hlc = dataframe[["high", "low", "close"]].mean(axis=1)
        high_smma = self._cal_smma(dataframe["high"], length_ma)
        low_smma = self._cal_smma(dataframe["low"], length_ma)
        middle_zlema = self._cal_zero_lag_ema(mean_hlc, length_ma)

        impulse_macd = np.where(middle_zlema > high_smma, middle_zlema - high_smma, 0)
        impulse_macd = np.where(middle_zlema < low_smma, middle_zlema - low_smma, impulse_macd)

        impulse_macd_signal = ta.SMA(impulse_macd, timeperiod=length_signal)
        impulse_macd_hist = impulse_macd - impulse_macd_signal

        return impulse_macd, impulse_macd_signal, impulse_macd_hist

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for ma_period in self.macd_ma_period.range:
            for signal_period in self.macd_signal_period.range:
                macd, macdsignal, macdhist = self.impulsive_macd(
                    dataframe, ma_period, signal_period
                )
                dataframe[f"impulse_macd_{ma_period}_{signal_period}"] = macd
                dataframe[f"impulse_macdsignal_{ma_period}_{signal_period}"] = macdsignal
                dataframe[f"impulse_macdhist_{ma_period}_{signal_period}"] = macdhist

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        dataframe["williams_r"] = ta.WILLR(dataframe, timeperiod=14)

        dataframe["cmf"] = self.chaikin_money_flow(dataframe, n=20, fillna=True)

        dataframe["volume_divergence"], dataframe["volume_divergence_signal"] = (
            self._cal_volume_divergence(dataframe, period=14)
        )

        inf_tf = "1d"
        informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=inf_tf)

        for ema_period in self.day_ema_period.range:
            informative[f"ema_{ema_period}"] = ta.EMA(informative["close"], period=ema_period)
            informative[f"ema_diff_{ema_period}"] = informative[f"ema_{ema_period}"].shift(1)

        inf_heikinashi = qtpylib.heikinashi(informative)

        informative["ha_rsi"] = ta.RSI(inf_heikinashi, timeperiod=14)
        informative["ha_williams_r"] = ta.WILLR(inf_heikinashi, timeperiod=14)

        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, inf_tf, ffill=True
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        triggers_long = []
        triggers_short = []

        guards_long = []
        guards_short = []

        triggers_long.append(
            qtpylib.crossed_above(
                dataframe[
                    f"impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
            )
        )
        triggers_short.append(
            qtpylib.crossed_below(
                dataframe[
                    f"impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
            )
        )

        if self.volume_enabled.value:
            triggers_long.append(
                qtpylib.crossed_above(
                    dataframe["volume_divergence"], dataframe["volume_divergence_signal"]
                )
            )
            triggers_short.append(
                qtpylib.crossed_below(
                    dataframe["volume_divergence"], dataframe["volume_divergence_signal"]
                )
            )

        if self.check_macd_position.value:
            guards_long.append(
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ]
                < 0
            )
            guards_short.append(
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ]
                > 0
            )

        guards_long.append(dataframe["volume"] > 0)
        guards_short.append(dataframe["volume"] > 0)

        if self.rsi_enabled.value:
            guards_long.append(dataframe["rsi"] > self.rsi_long_threshold.value)
            guards_short.append(dataframe["rsi"] < self.rsi_short_threshold.value)


        if self.ha_rsi_enabled.value:
            guards_long.append(dataframe["ha_rsi_1d"] < self.ha_rsi_long_threshold.value)
            guards_short.append(dataframe["ha_rsi_1d"] > self.ha_rsi_short_threshold.value)

        if self.williams_r_enabled.value:
            guards_long.append(dataframe["williams_r"] > self.williams_r_long_threshold.value)
            guards_short.append(dataframe["williams_r"] < self.williams_r_short_threshold.value)


        if self.ha_williams_r_enabled.value:
            guards_long.append(
                dataframe["ha_williams_r_1d"] < self.ha_williams_r_long_threshold.value
            )
            guards_short.append(
                dataframe["ha_williams_r_1d"] > self.ha_williams_r_short_threshold.value
            )

        if self.cmf_enabled.value:
            guards_long.append(dataframe["cmf"] > self.cmf_long_threshold.value)
            guards_short.append(dataframe["cmf"] < self.cmf_short_threshold.value)

        if self.day_ema_enabled.value:
            guards_long.append(
                (dataframe[f"ema_{self.day_ema_period.value}_1d"] > dataframe["close"])
                & (dataframe[f"ema_diff_{self.day_ema_period.value}_1d"] > 0)
            )
            guards_short.append(
                (dataframe[f"ema_{self.day_ema_period.value}_1d"] < dataframe["close"])
                & (dataframe[f"ema_diff_{self.day_ema_period.value}_1d"] < 0)
            )

        if triggers_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_long) & reduce(lambda x, y: x & y, guards_long),
                "enter_long",
            ] = 1

        if triggers_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_short)
                & reduce(lambda x, y: x & y, guards_short),
                "enter_short",
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        triggers_long = []
        triggers_short = []

        guards_long = []
        guards_short = []

        triggers_short.append(
            qtpylib.crossed_above(
                dataframe[
                    f"impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
            )
        )
        triggers_long.append(
            qtpylib.crossed_below(
                dataframe[
                    f"impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ],
            )
        )

        triggers_long.append(
            qtpylib.crossed_below(
                dataframe["volume_divergence"], dataframe["volume_divergence_signal"]
            )
        )
        triggers_short.append(
            qtpylib.crossed_above(
                dataframe["volume_divergence"], dataframe["volume_divergence_signal"]
            )
        )
        if self.check_macd_position.value:
            guards_short.append(
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ]
                < 0
            )
            guards_long.append(
                dataframe[
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}"
                ]
                > 0
            )

        guards_long.append(dataframe["volume"] > 0)
        guards_short.append(dataframe["volume"] > 0)

        if triggers_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_long) & reduce(lambda x, y: x & y, guards_long),
                "exit_long",
            ] = 1

        if triggers_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, triggers_short)
                & reduce(lambda x, y: x & y, guards_short),
                "exit_short",
            ] = 1

        return dataframe

    def chaikin_money_flow(self, dataframe, n=20, fillna=False) -> Series:
        """Chaikin Money Flow (CMF)
        It measures the amount of Money Flow Volume over a specific period.
        http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
        Args:
            dataframe(pandas.Dataframe): dataframe containing ohlcv
            n(int): n period.
            fillna(bool): if True, fill nan values.
        Returns:
            pandas.Series: New feature generated.
        """
        mfv = (
            (dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])
        ) / (dataframe["high"] - dataframe["low"])
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= dataframe["volume"]
        cmf = (
            mfv.rolling(n, min_periods=0).sum()
            / dataframe["volume"].rolling(n, min_periods=0).sum()
        )
        if fillna:
            cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
        return Series(cmf, name="cmf")

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:

        return 5

    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "IMPULSE_MACD": {
                    f"impulse_macd_{self.macd_ma_period.value}_{self.macd_signal_period.value}": {
                        "color": "blue"
                    },
                    f"impulse_macdsignal_{self.macd_ma_period.value}_{self.macd_signal_period.value}": {
                        "color": "orange"
                    },
                    f"impulse_macdhist_{self.macd_ma_period.value}_{self.macd_signal_period.value}": {
                        "type": "bar",
                        "plotly": {"opacity": 0.9},
                    },
                }
            },
        }
