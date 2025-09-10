# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TheForceV4(IStrategy):
  
    INTERFACE_VERSION = 2
    
    #V4 added roi_multiplier if you want to change the roi to n times the original value.
    #Based on Kage Raken idea and testing.
    
    roi_multiplier = 3
    
    minimal_roi = {
        "30": 0.005 * roi_multiplier,
        "15": 0.01 * roi_multiplier,
        "0": 0.012 * roi_multiplier
    }

    
    #V2 changed stoploss to 10%
    #stoploss = -0.03 #V1 original stoploss
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30


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
        
        # Momentum Indicators
        # ------------------------------------
        
        # Stochastic Slow added in V3 thanks to the idea of JoeSchr
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # MACD
        macd = ta.MACD(dataframe, 12, 26, 1)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # EMA - Exponential Moving Average
        dataframe['ema5h'] = ta.EMA(dataframe['high'], timeperiod=5)
        dataframe['ema5l'] = ta.EMA(dataframe['low'], timeperiod=5)
        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema5o'] = ta.EMA(dataframe['open'], timeperiod=5)

        # BOLLINGER
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
                (  # Original buy condition
                    ((dataframe['slowk'] >= 20) & (dataframe['slowk'] <= 80))
                    &
                    ((dataframe['slowd'] >= 20) & (dataframe['slowd'] <= 80))
                )
                &
                (
                    (
                        ( #Original buy condition
                            (dataframe['macd'] > dataframe['macd'].shift(1))
                            &
                            (dataframe['macdsignal'] > dataframe['macdsignal'].shift(1))
                        )
                        &
                        ( #Original buy condition
                            (dataframe['close'] > dataframe['close'].shift(1))
                        )
                        &
                        ( #Original buy condition
                            (dataframe['ema5c'] >= dataframe['ema5o'])
                            |
                            (dataframe['open'] < dataframe['ema5l'])
                        )
                    )
                    |
                    ( # V2 Added buy condition w/ Bollingers bands
                        (dataframe['close'] <= dataframe['bb_lowerband'])
                        |
                        (dataframe['open'] <= dataframe['bb_lowerband'])
                    )
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
                        (dataframe['slowk'] <= 80) & (dataframe['slowd'] <= 80)
                )
                &
                (
                    (
                        ( #Original buy condition
                            (dataframe['macd'] < dataframe['macd'].shift(1))
                            &
                            (dataframe['macdsignal'] < dataframe['macdsignal'].shift(1))
                        )
                        &
                        ( #Original buy condition
                            (dataframe['ema5c'] < dataframe['ema5o'])
                            |
                            (dataframe['open'] >= dataframe['ema5h']) ##V3 addon based on SmoothScalp
                        )
                    )
                    |
                    ( # V2 Added buy condition w/ Bollingers bands
                        (dataframe['close'] >= dataframe['bb_upperband'])
                        |
                        (dataframe['open'] >= dataframe['bb_upperband'])
                    )
                )
            ),
            'sell'] = 1
        return dataframe
    