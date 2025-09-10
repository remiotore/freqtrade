import logging

import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IntParameter, informative
from freqtrade.strategy.interface import IStrategy
from indicators import premium_stochastic_oscillator
from pandas import DataFrame

logger = logging.getLogger(__name__)


class pso(IStrategy):
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
    startup_candle_count: int = 32

    pso_length_buy = IntParameter(1, 60, default=32, space="buy")
    pso_length_sell = IntParameter(1, 60, default=32, space="sell")

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
                "pso": {"pso": {"color": "#26a269", "type": "line"}},
            },
        }

    def populate_pso_indicators(self, dataframe: DataFrame) -> DataFrame:
        df = dataframe.copy()

        frames = [df]

        for val in self.pso_length_buy.range:
            frames.append(premium_stochastic_oscillator(frames, period=val))

        for val in self.pso_length_sell.range:
            frames.append(
                frames.append(premium_stochastic_oscillator(frames, period=val))
            )

        return pd.concat(frames, axis=1)

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.populate_pso_indicators(dataframe)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.populate_pso_indicators(dataframe)
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
