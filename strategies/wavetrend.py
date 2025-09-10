


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series, DatetimeIndex, merge

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class wavetrend(IStrategy):
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
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """


    INTERFACE_VERSION = 2



    minimal_roi = {
        "0": 1
    }

    stoploss = -0.1

    trailing_stop = True

    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = True


    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

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
            'tema': {},
            'sar': {'color': 'white'},
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
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []


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


        self.n1 = 10 #WT Channel Length
        self.n2 = 21 #WT Average Length
        dataframe = self.market_cipher(dataframe)

        return dataframe


    
   
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (
                    dataframe['wtCrossUp'] &
                    dataframe['wtOversold']  
                )
                
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
               dataframe['wtCrossDown'] &
               dataframe['wtOverbought'] 

            ),
            'sell'] = 1

        return dataframe
        

    def market_cipher(self, dataframe) -> DataFrame:


        osLevel = -60
        obLevel = 30
        dataframe['ap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['esa'] = ta.EMA(dataframe['ap'], self.n1)
        dataframe['d'] = ta.EMA((dataframe['ap']-dataframe['esa']).abs(), self.n1)
        dataframe['ci'] = ( dataframe['ap']-dataframe['esa'] ) / (0.015 * dataframe['d'])
        dataframe['tci'] = ta.EMA(dataframe['ci'], self.n2)

        dataframe['wt1'] = dataframe['tci']
        dataframe['wt2'] = ta.SMA(dataframe['wt1'],4)
        
        dataframe['wtVwap'] = dataframe['wt1'] - dataframe['wt2']
        dataframe['wtOversold'] =   dataframe['wt2'] <= osLevel
        dataframe['wtOverbought'] =   dataframe['wt2'] >= obLevel
        

        dataframe['wtCrossUp'] = dataframe['wt2'] - dataframe['wt1'] <= 0
        dataframe['wtCrossDown'] = dataframe['wt2'] - dataframe['wt1'] >= 0
        dataframe['crossed_above'] = qtpylib.crossed_above(dataframe['wt2'], dataframe['wt1'])
        dataframe['crossed_below'] = qtpylib.crossed_below(dataframe['wt2'], dataframe['wt1'])
         
        return dataframe