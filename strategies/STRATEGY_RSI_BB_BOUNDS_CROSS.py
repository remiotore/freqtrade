

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


_trend_length = 14
_bb_smooth_length=4

def iff(a,b,c):
    if a:
        return b
    else:
        return c

class STRATEGY_RSI_BB_BOUNDS_CROSS(IStrategy):
    """
    Strategy RSI_BB_BOUNDS_CROSS
    author@: Fractate_Dev
    github@: https://github.com/Fractate/freqbot
    How to use it?
    > python3 ./freqtrade/main.py -s RSI_BB_BOUNDS_CROSS
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }


    stoploss = -0.10

    trailing_stop = False

    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 20

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
            'bb_ub': {'color': 'green'},
            'bb_lb': {'color': 'green'},
            'bb_lb_smoothed': {'color': 'red'},
            'rsi_ub': {'color': 'black'},
            'rsi_lb': {'color': 'black'},
            'rsi_lb_smoothed': {'color': 'orange'},
        },
        'subplots': {






























            "bb_rsi_count": {
                'ub_bb_over_rsi_trend': {},
                'lb_bb_under_rsi_trend': {},
            },
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

        for i in range(1):
            print("")




        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=14, stds=2)
        dataframe['bb_lb'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_ub'] = bollinger['upper']

        dataframe['bb_lb_smoothed'] = 0
        for i in range(_bb_smooth_length):
            dataframe['bb_lb_smoothed'] += dataframe['bb_lb'].shift(i)/_bb_smooth_length
        
        dataframe['bb_percent'] = (dataframe['close'] - bollinger['lower']) / (bollinger['upper'] - bollinger['lower'])

        dataframe['rsi'] = ta.RSI(dataframe) # rsi(prices, period \\ 14)
        dataframe['70'] = 70
        dataframe['30'] = 30
        dataframe['1'] = 1
        dataframe['0'] = 0
        
        rsi_limit = 30
        dataframe['rsi_percent'] = (dataframe['rsi'] - rsi_limit) / (100 - rsi_limit * 2)









        length = 14

        ep = 2 * length - 1

        dataframe.loc[(dataframe['close'] - dataframe['close'].shift(1)) > 0, 'auc1'] = (dataframe['close'] - dataframe['close'].shift(1))
        dataframe.loc[(dataframe['close'] - dataframe['close'].shift(1)) <= 0, 'auc1'] = 0
        dataframe['auc'] = ta.EMA(dataframe['auc1'], ep)

        dataframe.loc[(dataframe['close'].shift(1) - dataframe['close']) > 0, 'adc1'] = (dataframe['close'].shift(1) - dataframe['close'])
        dataframe.loc[(dataframe['close'].shift(1) - dataframe['close']) <= 0, 'adc1'] = 0
        dataframe['adc'] = ta.EMA(dataframe['adc1'], ep)

        dataframe['x1'] = (length - 1) * ( dataframe['adc'] * 70 / (100-70) - dataframe['auc'] )
        dataframe['x1'] = np.nan_to_num(dataframe['x1'])

        dataframe.loc[dataframe['x1'] >= 0, 'rsi_ub'] = dataframe['close'] + dataframe['x1']
        dataframe.loc[dataframe['x1'] < 0, 'rsi_ub'] = dataframe['close'] + dataframe['x1'] * (100-70)/70
        
        dataframe['x2'] = (length - 1) * ( dataframe['adc'] * 30 / (100-30) - dataframe['auc'] )
        dataframe.loc[dataframe['x2'] >= 0, 'rsi_lb'] = dataframe['close'] + dataframe['x2']
        dataframe.loc[dataframe['x2'] < 0, 'rsi_lb'] = dataframe['close'] + dataframe['x2'] * (100-30)/30
        
        dataframe['rsi_lb_smoothed'] = 0
        for i in range(_bb_smooth_length):
            dataframe['rsi_lb_smoothed'] += dataframe['rsi_lb'].shift(i)/_bb_smooth_length








        dataframe['bb_minus_rsi_percent'] = dataframe['bb_percent'] - dataframe['rsi_percent']

        dataframe['ub_bb_over_rsi'] = dataframe['bb_ub'] > dataframe['rsi_ub']
        dataframe['lb_bb_under_rsi'] = dataframe['bb_lb'] < dataframe['rsi_lb']

        dataframe['ub_bb_over_rsi_trend'] = True
        dataframe['lb_bb_under_rsi_trend'] = True



        for i in range(_trend_length):


            dataframe['ub_bb_over_rsi_trend'] = dataframe['ub_bb_over_rsi'].shift(i) & dataframe['ub_bb_over_rsi_trend']
            dataframe['lb_bb_under_rsi_trend'] = dataframe['lb_bb_under_rsi'].shift(i) & dataframe['lb_bb_under_rsi_trend']

        

        dataframe['ema14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['ema2'] = ta.EMA(dataframe, timeperiod=2)
















        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (






                (dataframe['lb_bb_under_rsi_trend']) &




                (dataframe['bb_lb'] < dataframe['rsi_lb']) &
                (dataframe['bb_lb'].shift(1) < dataframe['rsi_lb'].shift(1)) &




                (dataframe['ema2'] > dataframe['ema2'].shift(1))



            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
















                (dataframe['ema2'] < dataframe['ema2'].shift(1))
            ),
            'sell'] = 1
        return dataframe
    