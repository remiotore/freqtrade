



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from functools import reduce

class PRICEFOLLOWINGX(IStrategy):
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
    @property
    def protections(self):
            return [
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": 48,
                    "trade_limit": 5,
                    "stop_duration_candles": 5,
                    "max_allowed_drawdown": 0.75
                },
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 24,
                    "trade_limit": 3,
                    "stop_duration_candles": 5,
                    "only_per_pair": True
                },
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": 30,
                    "trade_limit": 2,
                    "stop_duration_candles": 6,
                    "required_profit": 0.005
                },
            ]


    INTERFACE_VERSION = 2


    minimal_roi = {
        "120":0.015,
        "60": 0.025,
        "30": 0.03,
        "0": 0.015
       }


    stoploss = -0.5

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03  # Disabled / not configured

    rsi_enabled = BooleanParameter(default=True, space='buy', optimize=True, load=True)

    ema_pct = DecimalParameter(0.001, 0.100, decimals = 3, default = 0.040, space="buy", optimize=True, load=True)
    buy_frsi = DecimalParameter(-0.71, 0.50, decimals = 2, default = -0.40, space="buy", optimize=True, load=True)
    frsi_pct = DecimalParameter(0.01, 0.20, decimals = 2, default = 0.10, space="buy", optimize=True, load=True)

    ema_sell_pct = DecimalParameter(0.001, 0.020, decimals = 3, default = 0.003, space="sell", optimize=True, load=True)
    sell_rsi_enabled = BooleanParameter(default=True, space='sell', optimize=True, load=True)
    sell_frsi = DecimalParameter(-0.30, 0.70, decimals=2, default=0.2, space="sell", load=True)

    timeframe = '15m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 20

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
            'ema7low':{},
            'ema10high':{},
            'ha_open':{},
            'ha_close':{},
        },
        'subplots': {




            "RSI": {
                'frsi': {'color': 'red'},
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
        return [("ETH/BUSD", "1h"),
                ("LINK/BUSD", "1h"),
                ("RVN/BUSD", "1h"),
                ("MATIC/BUSD", "30m")
                        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe, window=14)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['frsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)







        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']



        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=19, stds=2.2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=7)









        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['ema7'] = ta.SMA(dataframe, timeperiod=14)
        dataframe['emalow'] = ta.EMA(dataframe, timeperiod=12, price='low')
        dataframe['emahigh'] = ta.EMA(dataframe, timeperiod=14, price='high')


        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        last_emalow = dataframe['emalow'].tail()
        last_tema = dataframe['tema'].tail()
        haclose = dataframe['ha_close'].tail(3)
        haclose3rdlast, haclose2ndlast, hacloselast = haclose
        haopen = dataframe['ha_open']
        Conditions = []

        if self.rsi_enabled.value:
           Conditions.append(qtpylib.crossed_below(dataframe['frsi'], self.buy_frsi.value))
           Conditions.append(dataframe['tema'] < dataframe['bb_lowerband'])
           Conditions.append(qtpylib.crossed_below(dataframe['tema'], dataframe['emalow']))

            
        else:
           Conditions.append(dataframe['tema'] > dataframe['bb_middleband'])
           Conditions.append(qtpylib.crossed_above(dataframe['tema'], dataframe['ema7']))


        
        if Conditions:
             dataframe.loc[
                 reduce(lambda x, y: x & y, Conditions),
                 'buy'] = 1

        return dataframe







    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

            haopen = dataframe['ha_open']
            haclose = dataframe['ha_close']
            last_tema = dataframe['tema'].tail()
            last_emahigh = dataframe['emahigh'].tail()
            conditions = []

            if self.sell_rsi_enabled.value:
                 conditions.append(qtpylib.crossed_below(dataframe['frsi'], self.sell_frsi.value))
                 conditions.append(dataframe['tema'] < dataframe['bb_middleband'])

                 conditions.append(qtpylib.crossed_below(dataframe['tema'], dataframe['ema7']))

            else:
                 conditions.append(dataframe['tema'] < dataframe['bb_middleband'])
                 conditions.append(qtpylib.crossed_below(dataframe['tema'], dataframe['ema7']))



            if conditions:
                 dataframe.loc[
                      reduce(lambda x, y: x & y, conditions),
                      'sell'] = 1

            return dataframe
