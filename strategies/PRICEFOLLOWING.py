



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce

class PRICEFOLLOWING(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
        "60": 0.025,
        "30": 0.03,
        "0": 0.04
    }


    stoploss = -0.1

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03  # Disabled / not configured

    rsi_value = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    rsi_enabled = BooleanParameter(default=False, space='buy', optimize=True, load=True)
    ema_pct = DecimalParameter(0.0001, 0.1, decimals = 4, default = 0.004, space="buy", optimize=True)

    ema_sell_pct = DecimalParameter(0.0001, 0.1, decimals = 4, default = 0.003, space="sell", optimize=True, load=True)
    sell_rsi_value = IntParameter(low=25, high=100, default=70, space='sell', optimize=True, load=True)
    sell_rsi_enabled = BooleanParameter(default=True, space='sell', optimize=True, load=True)

    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 15

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'ema7':{},
            'ha_open':{},
            'ha_close':{},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
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
            
        """
        return [("ETH/USDT", "15m"),
                ("BTC/USDT", "15m"),
                ("RVN/USDT", "15m")
                        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)










        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema24'] = ta.EMA(dataframe, timeperiod=24)



        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=7)



        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']



        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        last_ema7 = dataframe['ema7'].tail()
        last_tema = dataframe['tema'].tail()
        haclose = dataframe['ha_close'].tail(4)
        haclose4thlast, haclose3rdlast, haclose2ndlast, hacloselast = haclose
        haopen = dataframe['ha_open']
        Conditions = []

        if self.rsi_enabled.value:
                Conditions.append(dataframe['rsi'] < self.rsi_value.value)
                Conditions.append(qtpylib.crossed_below(dataframe['ema7'], dataframe['tema']))
        else:
            Conditions.append(qtpylib.crossed_below(dataframe['ema7'], dataframe['tema']))

            Conditions.append(dataframe['tema'] < dataframe['tema'].shift(1))


       
        if Conditions:
             dataframe.loc[
                 reduce(lambda x, y: x & y, Conditions),
                 'buy'] = 1
       
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            
            haopen = dataframe['ha_open']
            haclose = dataframe['ha_close']
            last_ema7 = dataframe['ema7'].tail()
            last_tema = dataframe['tema'].tail()
            haclose = dataframe['ha_close'].tail(2)
            haclose2ndlast, hacloselast = haclose
            conditions = []

            if self.sell_rsi_enabled.value:
                conditions.append(dataframe['rsi'] < self.sell_rsi_value.value)
                conditions.append(qtpylib.crossed_above(dataframe['ema7'], dataframe['tema']))
            else:

                conditions.append(qtpylib.crossed_above(dataframe['ema7'], dataframe['tema']))



            if conditions:
                 dataframe.loc[
                      reduce(lambda x, y: x & y, conditions),
                      'sell'] = 1

            return dataframe
