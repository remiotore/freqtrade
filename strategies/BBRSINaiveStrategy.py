


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class BBRSINaiveStrategy(IStrategy):


    INTERFACE_VERSION = 2


    minimal_roi = {


        "0": 100

    }


    stoploss = -0.8

    trailing_stop = False




    timeframe = '4h'



    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 200

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

    plot_config = {
        'main_plot': {

            'sma50': {'color': 'red'},
            'sma100': {},
            'ema21': {'color': 'green'},
            'ema50': {'color': 'orange'},
            'ema100': {'color': 'pink'},
            'ema150': {'color': 'brown'},
            'ema200': {'color': 'purple'},
            'BBANDS_U': {},
            'BBANDS_M': {},
            'BBANDS_L': {},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'yellow'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=15)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_midband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']

        bb = ta.BBANDS(dataframe, window=20, stds=2)


        dataframe['BBANDS_U'] = bb["upperband"]
        dataframe['BBANDS_M'] = bb["middleband"]
        dataframe['BBANDS_L'] = bb["lowerband"]

        weighted_bollinger = qtpylib.weighted_bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        dataframe["wbb_percent"] = (
                (dataframe["close"] - dataframe["wbb_lowerband"]) /
                (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        )
        dataframe["wbb_width"] = (
                (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) /
                dataframe["wbb_middleband"]
        )

        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)


        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema150'] = ta.EMA(dataframe, timeperiod=150)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        dataframe["rsi_buy_hline"] = 25
        dataframe["rsi_sell_hline"] = 60

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 25)  # Signal: RSI is greater 25
                    & qtpylib.crossed_below(dataframe['sma50'], dataframe['ema21'])
                    & (dataframe['close'] > dataframe['ema100'])


            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                qtpylib.crossed_above(dataframe['sma50'], dataframe['ema21'])

            ),
            'sell'] = 1

        return dataframe
