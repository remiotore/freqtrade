from datetime import datetime, timedelta
import pandas_ta as pta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class FSupertrendStrategy(IStrategy):
    can_short = True

    INTERFACE_VERSION: int = 3
    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 2,
        "buy_m2": 4
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 5,
        "sell_profit": 0.005,
        "sell_fastk_min": 5,
        "sell_fastk_max": 95,
        "sell_profit2": -0.008,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.02,
        "8": 0.015,
        "14": 0.01,
        "18": 0.006,
        "20": 0.001
    }

    # Stoploss:
    stoploss = -0.025

    # Trailing stop:
    trailing_stop = True                   # 启用追踪止损
    trailing_stop_positive = 0.0025         # 激活后，止损位设置在当前价格峰值下方 0.2%
    trailing_stop_positive_offset = 0.0065  # 当盈利达到 0.5% 时激活追踪止损       
    trailing_only_offset_is_reached = True # 仅在盈利达到 0.5% 后才开始追踪

    timeframe = "1m"

    startup_candle_count = 80

    buy_m1 = IntParameter(1, 8, default=1, optimize=False)
    buy_m2 = IntParameter(1, 8, default=1, optimize=False)

    buy_rsi_min = IntParameter(10, 30, default=20, optimize=False)
    buy_rsi_max = IntParameter(70, 90, default=80, optimize=False)

    sell_m1 = IntParameter(1, 5, default=1, optimize=False)

    sell_profit = DecimalParameter(0, 0.01, default=0.003, optimize=False)
    sell_profit2 = DecimalParameter(-0.015, -0.002, default=-0.008, optimize=True)
    sell_fastk_max = IntParameter(80, 100, default=90, optimize=False)
    sell_fastk_min = IntParameter(0, 20, default=10, optimize=False)

    # @property
    # def protections(self):
    # return  [
    # {
    # "method": "CooldownPeriod",
    # "stop_duration_candles": 1
    # }
    # ]

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, side, **kwargs):
        return 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # for multiplier in self.buy_m1.range:
        #     dataframe[f"supertrend_1_buy"] = supertrend(dataframe, multiplier, 5)["STX"]
        # for multiplier in self.buy_m2.range:
        #     dataframe[f"supertrend_2_buy"] = supertrend(dataframe, multiplier, 10)["STX"]

        # for multiplier in self.sell_m1.range:
        # dataframe[f"supertrend_1_sell"] = supertrend(dataframe, multiplier, 10)["STX"]

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=5)
        
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=10)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        
        dataframe['rsi_fastk'] = dataframe['rsi'] + dataframe['fastk']
        

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            # (dataframe[f"supertrend_1_buy"] == "up")

            (dataframe['macd'] < dataframe['macdsignal'])
            # & (dataframe['macdhist'] < dataframe['macdhist'].shift(1))
            & (dataframe['rsi'] < 15)

            & (dataframe['fastk'] < 5)

            & (dataframe['rsi_fastk'] < 15)
            & (dataframe['adx'] > 30)
            # & (dataframe['fastk'] < dataframe['fastk'].shift(1))

            & (dataframe["volume"] > 0),
            ["enter_short", "enter_tag"]
        ] = (1, "long")

        dataframe.loc[
            # (dataframe[f"supertrend_1_buy"] == "down")

            (dataframe['macd'] > dataframe['macdsignal'])
            # & (dataframe['macdhist'] > dataframe['macdhist'].shift(1))
            & (dataframe['rsi'] > 85)

            & (dataframe['fastk'] > 95)

            & (dataframe['rsi_fastk'] > 185)
            & (dataframe['adx'] > 30)
            # & (dataframe['fastk'] > dataframe['fastk'].shift(1))

            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"]
        ] = (1, "short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_profit > 0.005:
            if trade.is_short:
                if current_candle["fastk"] < 5:
                    return "fastk_exit_short"
            else:
                if current_candle["fastk"] > 95:
                    return "fastk_exit_long"


        if current_profit > 0.001 and (current_time - timedelta(minutes=12) > trade.open_date_utc):
            if trade.is_short:
                if current_candle["fastk"] == 0:
                    return "fastk_exit_short10"
            else:
                if current_candle["fastk"] == 100:
                    return "fastk_exit_long10"

        if current_profit < -0.008:
            if trade.is_short:
                if current_candle["fastk"] < 10:
                    return "fastk_stoploss_short"
            else:
                if current_candle["fastk"] > 90:
                    return "fastk_stoploss_long"
