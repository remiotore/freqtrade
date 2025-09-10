

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TheForceV7(IStrategy):
  
    INTERFACE_VERSION = 2


























    
    minimal_roi = {





		"0": 10 
    }


    stoploss = -0.1

    trailing_stop = False




    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

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



        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        dataframe['rsi7'] = ta.RSI(dataframe, timeperiod=7)

        macd = ta.MACD(dataframe,12,26,1)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['ema5h'] = ta.EMA(dataframe['high'], timeperiod=5)
        dataframe['ema5l'] = ta.EMA(dataframe['low'], timeperiod=5)
        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema5o'] = ta.EMA(dataframe['open'], timeperiod=5)
        dataframe['ema200c'] = ta.MA(dataframe['close'], 200)

        dataframe['volvar'] = (dataframe['volume'].rolling(100).mean() * 1.5)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=21, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        
        
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
                ( 
                    (
                        ( #Original buy condition
                            (dataframe['slowk'] >= 20) & (dataframe['slowk'] <= 80)
                            &
                            (dataframe['slowd'] >= 20) & (dataframe['slowd'] <= 80)
                        )
                        |
                        (  #V3 added based on SmoothScalp
                            (dataframe['slowk'] < 30) & (dataframe['slowd'] < 30) &
                            (qtpylib.crossed_above(dataframe['slowk'], dataframe['slowd']))
                        )
                    )
                    &
                    ( #Original buy condition #Might need improvement to have better signals
                        (dataframe['macd'] > dataframe['macd'].shift(1))
                        &
                        (dataframe['macdsignal'] > dataframe['macdsignal'].shift(1))
                    )
                    &
                    ( #Original buy condition
                        (dataframe['close'] > dataframe['close'].shift(1))
                        & #V6 added condition to improve buy's
                        (dataframe['open'] > dataframe['open'].shift(1)) 
                    )
                    &
                    ( #Original buy condition
                        (dataframe['ema5c'] >= dataframe['ema5o'])
                        |
                        (dataframe['open'] < dataframe['ema5l'])
                    )
                    &
                    (

                       (dataframe['volume'] > dataframe['volvar'])
                    )
                )
                |
                ( # V2 Added buy condition w/ Bollingers bands
                    (dataframe['slowk'] >= 20) & (dataframe['slowk'] <= 80)
                    &
                    (dataframe['slowd'] >= 20) & (dataframe['slowd'] <= 80)
                    &
                    (
                        (dataframe['close'] <= dataframe['bb_lowerband'])
                        |
                        (dataframe['open'] <= dataframe['bb_lowerband'])
                    )
                )
                |
                (  # V5 added Pullback RSI thanks to simoelmou
                    (dataframe['close'] > dataframe['ema200c']) 
					&
                    (dataframe['rsi7'] < 35)
                )
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
                (
                    (
                        ( #Original sell condition
                            (dataframe['slowk'] <= 80)  & (dataframe['slowd'] <= 80)
                        )
                        |
                        ( #V3 added based on SmoothScalp
                            (qtpylib.crossed_above(dataframe['slowk'], 70))
							|
                            (qtpylib.crossed_above(dataframe['slowd'], 70))
                        )
                    )
                    &
                    ( #Original sell condition
                        (dataframe['macd'] < dataframe['macd'].shift(1))
                        &
                        (dataframe['macdsignal'] < dataframe['macdsignal'].shift(1))
                    )
                    &
                    ( #Original sell condition
                        (dataframe['ema5c'] < dataframe['ema5o'])
                        |
                        (dataframe['open'] >= dataframe['ema5h']) # V3 added based on SmoothScalp
                    )
                )
                |
                ( # V2 Added sell condition w/ Bollingers bands
                    (dataframe['slowk'] <= 80)
                    &
                    (dataframe['slowd'] <= 80)
                    &
                    ( 
                        (dataframe['close'] >= dataframe['bb_upperband'])
                        |
                        (dataframe['open'] >= dataframe['bb_upperband'])
                    )
                )
                |
                (# V6 Added sell condition for extra high values
                    (dataframe['high'] > dataframe['bb_upperband'])
                    &
                    (((dataframe['high'] - dataframe['bb_upperband']) * 100 / dataframe['bb_upperband']) > 1)
                )
                
            ),
            'sell'] = 1
        return dataframe
    