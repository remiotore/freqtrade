import datetime

import pytz
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
import numpy as np
from freqtrade.strategy import IStrategy, informative


def calculate_angle(series, periods):
    radians = np.arctan((series.diff(periods) / periods).values)
    degrees = np.degrees(radians)
    return degrees


class sonicr_strategy(IStrategy):
    """
    SonicR strategy
    - Main timeframe: 4h
    - Context:
        + EMA34 > EMA89
        + 50 < RSI14 < 60
    - Signal:
        + Bullish candlestick pattern
        + Lowest price near ema34, close price > ema34
    """

    INTERFACE_VERSION = 3

    can_short: bool = False

    timeframe = "4h"
    ema: int = 34
    rsi: int = 14
    startup_candle_count: int = 200

    stoploss = -0.05

    bullish_candlestick_patterns = {















        "DOJI": ta.CDLDOJI,


        "ENGULFING": ta.CDLENGULFING,

        "EVENING STAR": ta.CDLEVENINGSTAR,


        "HAMMER": ta.CDLHAMMER,
        "HANGING MAN": ta.CDLHANGINGMAN,


















        "MORNING STAR": ta.CDLMORNINGSTAR,





        "SHOOTING STAR": ta.CDLSHOOTINGSTAR,











    }

    def informative_pairs(self):
        """
        - Get daily candles infor for all pairs
        """

        pairs = self.dp.current_whitelist()

        informative_pairs = [(pair, "1d") for pair in pairs]
        return informative_pairs

    @informative("1d")
    def populate_indicators_1d(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        -
        - price close > ema89  and (ema34 > ema89 or ema89 - ema34 < 5% of ema89)

        """

        dataframe["ema34"] = ta.EMA(dataframe, timeperiod=34)
        dataframe["ema89"] = ta.EMA(dataframe, timeperiod=89)

        for (
            pattern_name,
            pattern_function,
        ) in self.bullish_candlestick_patterns.items():
            dataframe[pattern_name] = pattern_function(dataframe)
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:

        dataframe["ema34"] = ta.EMA(dataframe, timeperiod=34)
        dataframe["ema89"] = ta.EMA(dataframe, timeperiod=89)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        for (
            pattern_name,
            pattern_function,
        ) in self.bullish_candlestick_patterns.items():
            dataframe[pattern_name] = pattern_function(dataframe)
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        """
        dataframe format
        - ema_trend: up, down
        - open, close, low, high
        - rsi_trend: up, down
        - price_high_trend: up, down
        - hammer_trend: up, down
        - candle_trend: up, down
        - enter_long: 1, 0

        """


        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema34_1d"])







            ),
            "daily_trend",
        ] = "up"

        dataframe.loc[
            (dataframe["ema34"] >= dataframe["ema89"]), "ema_trend"
        ] = "up"
        dataframe.loc[
            (dataframe["ema34"] < dataframe["ema89"]), "ema_trend"
        ] = "down"



        dataframe["ema34_angle"] = calculate_angle(dataframe["ema34"], 20)
        dataframe.loc[(dataframe["ema34_angle"] > 30), "ema34_angle_trend"] = (
            "up"
        )

        dataframe.loc[
            (dataframe["rsi"] > 50) & (dataframe["rsi"] < 65), "rsi_trend"
        ] = "up"
        dataframe.loc[
            (dataframe["rsi"] <= 50) | (dataframe["rsi"] >= 65), "rsi_trend"
        ] = "down"

        dataframe.loc[
            (dataframe["close"] > dataframe["open"]), "price_high_trend"
        ] = "up"

        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema34"])
                & (
                    (dataframe["low"] - dataframe["ema34"])
                    < (dataframe["open"] - dataframe["low"])
                )










            ),
            "hammer_trend",
        ] = "up"

        dataframe.loc[
            dataframe[list(self.bullish_candlestick_patterns.keys())]
            .gt(0)
            .any(axis=1),
            "candle_trend",
        ] = "up"

        dataframe.loc[
            (dataframe["ema_trend"] == "up")
            & (dataframe["rsi_trend"] == "up")
            & (dataframe["price_high_trend"] == "up")
            & (dataframe["hammer_trend"] == "up")
            & (dataframe["daily_trend"] == "up")
            & (dataframe["candle_trend"] == "up"),
            "enter_long",
        ] = 1

        if dataframe["enter_long"].iloc[-1] == 1:




            candle_names = []
            for pattern_name in self.bullish_candlestick_patterns.keys():
                if dataframe[pattern_name].iloc[-1] > 0:
                    candle_names.append(pattern_name)

            current_time = datetime.datetime.now(
                tz=pytz.timezone("Asia/Ho_Chi_Minh")
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.dp.send_msg(
                f"H4 - SonicR - {current_time}: Enter Long above price {dataframe['close'].iloc[-1]} for {metadata['pair']} with {', '.join(candle_names)} candles"
            )
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (

                dataframe["ema_trend"]
                == "down"
            ),
            "exit_long",
        ] = 1
        return dataframe
