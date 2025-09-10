# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce
# --------------------------------


class test(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 1,
    }

    stoploss = -0.05
    # --- Configuration ---
    timeframe = "1d"
    use_sell_signal = True
    sell_profit_only = False
    process_only_new_candles = True
    startup_candle_count = 10
    use_custom_stoploss = False
    # --- Configuration ---

    # storage dict for custom info
    custom_info = {}
    DATESTAMP = 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Check if the entry already exists
        if not metadata["pair"] in self.custom_info:
            # Create empty entry for this pair {datestamp, sellma, sell_trigger}
            self.custom_info[metadata["pair"]] = ['', 0, 0]
        dataframe['sell_custom'] = 0

        # EMA
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ema5'], dataframe['ema10']))
            ),
            ['buy', 'buy_tag']] = (1, 'Golden Cross')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell_custom'] = 0
        dataframe.loc[:, 'sell'] = 0
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["rsi"], 70))
            ),
            'sell_custom'] = 1

        return dataframe

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if(self.custom_info[pair][self.DATESTAMP] != last_candle['date']):
            self.custom_info[pair][self.DATESTAMP] = last_candle['date']
            if last_candle["sell_custom"] == 1:
                return "rsi_overbought"
        return None
