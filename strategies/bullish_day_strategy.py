import datetime

import pytz
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


class DailyCandleStrategy(IStrategy):
    """
    Alert when meet following conditions:
    - Bullish candlestick pattern
    - Close price > open price
    - abs(high - close) < abs(open - low) ~ hammer
    - Close price > EMA 34 and distance from low to EMA 34 < distance from open to low
    - Volume: TODO
    """

    INTERFACE_VERSION = 3

    can_short: bool = False

    timeframe = "1d"

    startup_candle_count: int = 14

    ema: int = 34
    rsi: int = 14

    stoploss = -0.05

    # https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html
    bullish_candlestick_patterns = {
        # "2 CROWS": ta.CDL2CROWS,
        # "3 BLACK CROWS": ta.CDL3BLACKCROWS,
        # "3 INSIDE": ta.CDL3INSIDE,
        # "3 LINE STRIKE": ta.CDL3LINESTRIKE,
        # "3 OUTSIDE": ta.CDL3OUTSIDE,
        # "3 STARS IN SOUTH": ta.CDL3STARSINSOUTH,
        # "3 WHITE SOLDIERS": ta.CDL3WHITESOLDIERS,
        # "ABANDONED BABY": ta.CDLABANDONEDBABY,
        # "ADVANCE BLOCK": ta.CDLADVANCEBLOCK,
        # "BELT HOLD": ta.CDLBELTHOLD,
        # "BREAKAWAY": ta.CDLBREAKAWAY,
        # "CLOSING MARUBOZU": ta.CDLCLOSINGMARUBOZU,
        # "CONCEAL BABY SWALL": ta.CDLCONCEALBABYSWALL,
        # "COUNTERATTACK": ta.CDLCOUNTERATTACK,
        # "DARK CLOUD COVER": ta.CDLDARKCLOUDCOVER,
        "DOJI": ta.CDLDOJI,
        # "DOJI STAR": ta.CDLDOJISTAR,
        # "DRAGONFLY DOJI": ta.CDLDRAGONFLYDOJI,
        "ENGULFING": ta.CDLENGULFING,
        # "EVENING DOJI STAR": ta.CDLEVENINGDOJISTAR,
        "EVENING STAR": ta.CDLEVENINGSTAR,
        # "GAP SIDE SIDE WHITE": ta.CDLGAPSIDESIDEWHITE,
        # "GRAVESTONE DOJI": ta.CDLGRAVESTONEDOJI,
        "HAMMER": ta.CDLHAMMER,
        "HANGING MAN": ta.CDLHANGINGMAN,
        # "HARAMI": ta.CDLHARAMI,
        # "HARAMI CROSS": ta.CDLHARAMICROSS,
        # "HIGH WAVE": ta.CDLHIGHWAVE,
        # "HIKKAKE": ta.CDLHIKKAKE,
        # "HIKKAKE MOD": ta.CDLHIKKAKEMOD,
        # "HOMING PIGEON": ta.CDLHOMINGPIGEON,
        # "IDENTICAL 3 CROWS": ta.CDLIDENTICAL3CROWS,
        # "IN NECK": ta.CDLINNECK,
        # "INVERTED HAMMER": ta.CDLINVERTEDHAMMER,
        # "KICKING": ta.CDLKICKING,
        # "KICKING BY LENGTH": ta.CDLKICKINGBYLENGTH,
        # "LADDER BOTTOM": ta.CDLLADDERBOTTOM,
        # "LONG LEGGED DOJI": ta.CDLLONGLEGGEDDOJI,
        # "LONG LINE": ta.CDLLONGLINE,
        # "MARUBOZU": ta.CDLMARUBOZU,
        # "MATCHING LOW": ta.CDLMATCHINGLOW,
        # "MAT HOLD": ta.CDLMATHOLD,
        # "MORNING DOJI STAR": ta.CDLMORNINGDOJISTAR,
        "MORNING STAR": ta.CDLMORNINGSTAR,
        # "ON NECK": ta.CDLONNECK,
        # "PIERCING": ta.CDLPIERCING,
        # "RICKSHAW MAN": ta.CDLRICKSHAWMAN,
        # "RISE FALL 3 METHODS": ta.CDLRISEFALL3METHODS,
        # "SEPARATING LINES": ta.CDLSEPARATINGLINES,
        "SHOOTING STAR": ta.CDLSHOOTINGSTAR,
        # "SHORT LINE": ta.CDLSHORTLINE,
        # "SPINNING TOP": ta.CDLSPINNINGTOP,
        # "STALLED PATTERN": ta.CDLSTALLEDPATTERN,
        # "STICK SANDWICH": ta.CDLSTICKSANDWICH,
        # "TAKURI": ta.CDLTAKURI,
        # "TASUKI GAP": ta.CDLTASUKIGAP,
        # "THRUSTING": ta.CDLTHRUSTING,
        # "TRISTAR": ta.CDLTRISTAR,
        # "UNIQUE 3 RIVER": ta.CDLUNIQUE3RIVER,
        # "UPSIDE GAP 2 CROWS": ta.CDLUPSIDEGAP2CROWS,
        # "XSIDE GAP 3 METHODS": ta.CDLXSIDEGAP3METHODS,
    }

    def informative_pairs(self):
        return []

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi)

        # EMA
        dataframe["ema_34"] = ta.EMA(dataframe, timeperiod=self.ema)

        # pattern recognition - bullish candlestick
        for (
            pattern_name,
            pattern_function,
        ) in self.bullish_candlestick_patterns.items():
            dataframe[pattern_name] = pattern_function(dataframe)
        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:

        # close price > open price
        dataframe.loc[
            (dataframe["close"] > dataframe["open"]), "price_high_trend"
        ] = "up"

        dataframe.loc[
            (dataframe["close"] <= dataframe["open"]), "price_high_trend"
        ] = "down"

        # rsi > 70 -> down
        dataframe.loc[(dataframe["rsi"] > 70), "rsi_trend"] = "down"

        # Bullish candlestick pattern
        dataframe.loc[
            dataframe[list(self.bullish_candlestick_patterns.keys())]
            .gt(0)
            .any(axis=1),
            "candle_trend",
        ] = "up"

        # Close price > EMA 34 and distance from low to EMA 34 < distance from open to low
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_34"])
                & (
                    (dataframe["low"] - dataframe["ema_34"])
                    < (dataframe["open"] - dataframe["low"])
                )
                # & (
                #     (
                #         (dataframe["high"] - dataframe["close"])
                #         < (dataframe["open"] - dataframe["low"])
                #     )
                #     or (
                #         (dataframe["high"] - dataframe["close"])
                #         < (dataframe["close"] - dataframe["open"])
                #     )
                # )
            ),
            "ema_trend",
        ] = "up"

        # combine all trend -> signal
        dataframe.loc[
            (dataframe["price_high_trend"] == "up")
            & (dataframe["candle_trend"] == "up")
            & (dataframe["ema_trend"] == "up"),
            "enter_long",
        ] = 1

        # notify signal if previsous candle have enter_long = 1
        if dataframe["enter_long"].iloc[-1] == 1:
            # get candle names for notification for current candle
            candle_names = []
            for pattern_name in self.bullish_candlestick_patterns.keys():
                if dataframe[pattern_name].iloc[-1] > 0:
                    candle_names.append(pattern_name)

            # Get current time in timezone HoChiMinh
            current_time = datetime.datetime.now(
                tz=pytz.timezone("Asia/Ho_Chi_Minh")
            ).strftime("%Y-%m-%d %H:%M:%S")
            self.dp.send_msg(
                f"Daily - {current_time}: Enter Long for {metadata['pair']} with {', '.join(candle_names)} candlestick pattern"
            )
        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        dataframe.loc[
            (
                # RSI crosse above 80
                (qtpylib.crossed_above(dataframe["rsi"], 80))
            ),
            "exit_long",
        ] = 1
        return dataframe
