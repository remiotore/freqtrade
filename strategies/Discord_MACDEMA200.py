import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class MACDEMA200(IStrategy):

    minimal_roi = {"0": 0}
    timeframe = "5m"
    stoploss = -1

    plot_config = {
        "main_plot": {
            "ema200": {},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Moving Average Convergence Divergence (MACD)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        # Exponential Moving Average (EMA)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        dataframe['cust_buy'] = (dataframe['macd'] < 0)
        dataframe.loc[(dataframe["cust_buy"] == True), "buy"] = 1

        # Buy Signal
        if (
            qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"])
            and dataframe["macd"] < 0
            and dataframe["close"] > dataframe["ema200"]
        ):
            dataframe["cust_stop"] = dataframe["low"].rolling(5).min()
            dataframe["cust_target"] = dataframe["close"] + (
                dataframe["close"] - dataframe["cust_stop"] * 1.5
            )
            dataframe["cust_buy"] = 1
        else:
            dataframe["cust_stop"] = dataframe["cust_stop"].shift(1)
            dataframe["cust_target"] = dataframe["cust_target"].shift(1)

        # Sell Signal
        if (
            qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])
            and dataframe["macd"] > 0
            and dataframe["close"] < dataframe["ema200"]
        ):
            dataframe["cust_sell"] = 1

        # Profit Target
        if dataframe["close"] > dataframe["cust_target"]:
            dataframe["cust_sell"] = 1

        # Stop Loss
        if dataframe["close"] < dataframe["cust_stop"]:
            dataframe["cust_sell"] = 1

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Signal: MACD Crossed Above MACD Signal
                # qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"])
                # Signal: MACD Below 0
                # & dataframe["macd"] < 0
                # Signal: Close Above EMA200
                # & dataframe["close"] > dataframe["ema200"]
                (dataframe["cust_buy"] == 1)
            ),
            "buy",
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Signal: MACD Crossed Below MACD Signal
                # qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])
                # Signal: MACD Above 0
                # & dataframe["macd"] > 0
                # Signal: Close Below EMA200
                # & dataframe["close"] < dataframe["ema200"]
                (dataframe["cust_sell"] == 1)
            ),
            "sell",
        ] = 1

        return dataframe
