from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

class 1231231231231234sfaf(IStrategy):

    buy_params = {
        "buy_fast": 5,
        "buy_slow": 20,
        "buy_push": 1.015,
        "buy_shift": -2,
        "buy_atr_multiplier": 1.5,
    }

    sell_params = {
        "sell_fast": 20,
        "sell_slow": 50,
        "sell_push": 0.98,
        "sell_shift": -2,
        "sell_atr_multiplier": 1.5,
    }

    minimal_roi = {
        "0": 0.02,
        "20": 0.01,
        "50": 0,
    }

    stoploss = -0.03

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    atr_period = 14

    timeframe = "5m"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["buy_ema_fast"] = ta.EMA(dataframe, timeperiod=self.buy_params["buy_fast"])
        dataframe["buy_ema_slow"] = ta.EMA(dataframe, timeperiod=self.buy_params["buy_slow"])
        dataframe["sell_ema_fast"] = ta.EMA(dataframe, timeperiod=self.sell_params["sell_fast"])
        dataframe["sell_ema_slow"] = ta.EMA(dataframe, timeperiod=self.sell_params["sell_slow"])

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            qtpylib.crossed_above(
                dataframe["buy_ema_fast"].shift(self.buy_params["buy_shift"]),
                dataframe["buy_ema_slow"].shift(self.buy_params["buy_shift"])
                * self.buy_params["buy_push"],
            )
        )

        conditions.append(dataframe["close"] > dataframe["buy_ema_slow"])
        conditions.append(dataframe["close"] > dataframe["buy_ema_fast"])
        conditions.append(
            dataframe["close"]
            > (dataframe["buy_ema_slow"] + dataframe["atr"] * self.buy_params["buy_atr_multiplier"])
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            qtpylib.crossed_below(
                dataframe["sell_ema_fast"].shift(self.sell_params["sell_shift"]),
                dataframe["sell_ema_slow"].shift(self.sell_params["sell_shift"])
                * self.sell_params["sell_push"],
            )
        )

        conditions.append(dataframe["close"] < dataframe["sell_ema_slow"])
        conditions.append(dataframe["close"] < dataframe["sell_ema_fast"])
        conditions.append(
            dataframe["close"]
            < (dataframe["sell_ema_slow"] - dataframe["atr"] * self.sell_params["sell_atr_multiplier"])
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "sell"] = 1

        return dataframe
