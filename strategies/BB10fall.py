


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BB10fall(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '1h'


    minimal_roi = {
        "0": 0.10
    }





    stoploss = -0.5

    trailing_stop = False




    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    @property
    def plot_config(self):
        return {

            'main_plot': {
                'bb_upperband': {'color': 'grey'},
                'bb_middle': {'color': 'red'},
                'bb_lowerband': {'color': 'grey'},
            },
            'subplots': {

                "RSI": {
                    'rsi': {'color': 'blue'},
                    'overbought': {'color': 'red'},
                    'oversold': {'color': 'green'},
                }
            }
        }

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:


        df['SMA 10'] = df["close"].rolling(window=10).mean()

        df['SD of SMA 10'] = df["SMA 10"].rolling(window=10).std()

        df['BB10 Upper'] = df["SMA 10"] + 2 * df['SD of SMA 10']

        df['BB10 Lower'] = df["SMA 10"] - 2 * df['SD of SMA 10']

        buyThreshold = (df['SMA 10'] + df['BB10 Lower'])/2
        buyThreshold = (buyThreshold + df['BB10 Lower'])/2
        buyThreshold = (buyThreshold + df['BB10 Lower'])/2
        df["buyThreshold10"] = buyThreshold

        sellThreshold = (df['SMA 10'] + df['BB10 Upper'])/2
        sellThreshold = (sellThreshold + df['BB10 Upper'])/2
        sellThreshold = (sellThreshold + df['BB10 Upper'])/2
        df["sellThreshold10"] = sellThreshold


        theEMAs = [5, 10, 12, 20, 26, 35, 50, 100]

        for x in theEMAs:
            df[f'EMA {x}'] = df["close"].ewm(
                span=x, min_periods=0, adjust=False, ignore_na=False).mean()

            index_no = df.columns.get_loc(f'EMA {x}')
            df.iloc[0: (x-1), [index_no]] = np.nan


        df["Price AvgOfInt"] = (df["open"] + df["close"]) / 2

        def fallOrRise(b, theVal, tIndex):
            if tIndex >= 4:
                if b == "Avg":
                    df_temp = df['Price AvgOfInt']
                else:
                    df_temp = df['close']

                if ((df_temp[df.index[tIndex]] > df_temp[df.index[tIndex-1]]) and
                    (df_temp[df.index[tIndex-1]] >
                             df_temp[df.index[tIndex-2]])
                    ):
                    value = "rise2"
                elif ((df_temp[df.index[tIndex]] < df_temp[df.index[tIndex-1]]) and
                      (df_temp[df.index[tIndex-1]] <
                       df_temp[df.index[tIndex-2]])
                      ):
                    value = "fall2"
                else:
                    value = ""
                return value

            else:
                value = ""
            return value

        AorC = ["Avg", "Clo"]

        for b in AorC:
            df[f"FallorRise for2d {b}Int"] = [fallOrRise(b, theVal, tIndex)
                                              for tIndex, (theVal)
                                              in enumerate(df['Price AvgOfInt'])]

        df["AvgOfVolume10Int"] = (
            df['volume'].rolling(window=10).mean().round(0))


        """

        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['close'] < df["buyThreshold10"]) &
                (df['close'].shift(1) < df["buyThreshold10"].shift(1)) &
                ((df["FallorRise for2d AvgInt"].shift(1) == "fall2") &
                    (df["FallorRise for2d AvgInt"].shift(2) == "fall2") &
                    (df["FallorRise for2d AvgInt"] == "")) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[
            (
                (df['close'] > df["EMA 5"]) &
                (df["AvgOfVolume10Int"] < df["volume"]) &
                (df['close'] > df["sellThreshold10"]) &
                (df['close'].shift(1) > df["sellThreshold10"].shift(1)) &
                ((df["FallorRise for2d AvgInt"].shift(1) == "rise2") &
                    (df["FallorRise for2d AvgInt"].shift(2) == "rise2") &
                    (df["FallorRise for2d AvgInt"] == "")) &
                (df['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1

        return df




