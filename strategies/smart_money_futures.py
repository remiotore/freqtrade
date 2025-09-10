import logging

import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import CategoricalParameter, IntParameter, informative
from freqtrade.strategy.interface import IStrategy
from indicators import smart_money_index, smi_trend
from mixins import SafetyOrderMixin, TrailingTakeProfitMixin
from pandas import DataFrame

logger = logging.getLogger(__name__)


class smart_money_futures(IStrategy):
    INTERFACE_VERSION = 3

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    stoploss = -0.99  # value loaded from strategy
    """
    END HYPEROPT
    """

    can_short: bool = True

    use_exit_signal: bool = True
    exit_profit_only: bool = False
    exit_profit_offset: float = 0.01
    ignore_roi_if_entry_signal: bool = True

    timeframe: str = "5m"

    process_only_new_candles: bool = True
    startup_candle_count: int = 200

    smart_money_index_fast_ma_long = IntParameter(10, 100, default=20, space="buy")
    smart_money_index_slow_ma_long = IntParameter(100, 350, default=150, space="buy")
    smart_money_index_fast_ma_short = IntParameter(10, 100, default=50, space="sell")
    smart_money_index_slow_ma_short = IntParameter(100, 300, default=150, space="sell")

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
                    f"smart_money_index_fast_ma_long_{self.smart_money_index_fast_ma_long.value}": {
                        "color": "#a51d2d",
                        "type": "line",
                    },
                },
                "smart_money_index_slow": {
                    "smart_money_index_1h": {"color": "#26a269", "type": "line"},
                    f"smart_money_index_slow_ma_long_{self.smart_money_index_slow_ma_long.value}_1h": {
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

        for val in self.smart_money_index_slow_ma_long.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_slow_ma_long_{val}": ta.SMA(
                            df["smart_money_index"], val
                        )
                    }
                )
            )

        for val in self.smart_money_index_fast_ma_long.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_fast_ma_long_{val}": ta.SMA(
                            df["smart_money_index"], val
                        )
                    }
                )
            )

        for val in self.smart_money_index_slow_ma_short.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_slow_ma_short_{val}": ta.SMA(
                            df["smart_money_index"], val
                        )
                    }
                )
            )

        for val in self.smart_money_index_fast_ma_short.range:
            frames.append(
                DataFrame(
                    {
                        f"smart_money_index_fast_ma_short_{val}": ta.SMA(
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
        dataframe.loc[
            (
                qtpylib.crossed_above(
                    dataframe["smart_money_index"],
                    dataframe[
                        f"smart_money_index_slow_ma_long_{self.smart_money_index_slow_ma_long.value}"
                    ],
                )
                & (dataframe["volume"] > 0)
                & (dataframe["smi_trend_1h"] != 2)
                & (dataframe["close"] < dataframe["ema50_1h"])
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "smart_money_long")

        dataframe.loc[
            (
                qtpylib.crossed_below(
                    dataframe["smart_money_index"],
                    dataframe[
                        f"smart_money_index_slow_ma_short_{self.smart_money_index_fast_ma_short.value}"
                    ],
                )
                & (dataframe["volume"] > 0)
                & (dataframe["smi_trend_1h"] == 2)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "smart_money_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


class SmartMoneyFutures_DCA(SafetyOrderMixin, SmartMoneyFutures):
    pass


class SmartMoneyFutures_DCA_TTP(
    TrailingTakeProfitMixin, SafetyOrderMixin, SmartMoneyFutures
):
    pass
