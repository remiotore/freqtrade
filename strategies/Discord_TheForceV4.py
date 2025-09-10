# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# Buy hyperspace params:
buy_params = {
    'bb_buy_size': 23,
    'ema_buy_size': 16,
    'macd_buy_size': 10,
    'slowd_buy_bottom': 12,
    'slowd_buy_up': 75,
    'slowk_buy_bottom': 11,
    'slowk_buy_up': 80
}

# Sell hyperspace params:
sell_params = {
    'bb_sell_size': 21,
    'ema_sell_size': 18,
    'macd_sell_size': 10,
    'slowd_sell': 85,
    'slowk_sell': 80
}

# Stoploss:
stoploss = -0.05


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
    stoploss = stoploss

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
        macd_buy = ta.MACD(dataframe, buy_params['macd_buy_size'], buy_params['macd_buy_size'] * 2, 1)
        dataframe['macd_buy'] = macd_buy['macd']
        dataframe['macdsignal_buy'] = macd_buy['macdsignal']

        macd_sell = ta.MACD(dataframe, sell_params['macd_sell_size'], sell_params['macd_sell_size'] * 2, 1)
        dataframe['macd_sell'] = macd_sell['macd']
        dataframe['macdsignal_sell'] = macd_sell['macdsignal']

        # EMA - Exponential Moving Average
        dataframe['emah_buy'] = ta.EMA(dataframe['high'], timeperiod=buy_params['ema_buy_size'])
        dataframe['emal_buy'] = ta.EMA(dataframe['low'], timeperiod=buy_params['ema_buy_size'])
        dataframe['emac_buy'] = ta.EMA(dataframe['close'], timeperiod=buy_params['ema_buy_size'])
        dataframe['emao_buy'] = ta.EMA(dataframe['open'], timeperiod=buy_params['ema_buy_size'])

        dataframe['emah_sell'] = ta.EMA(dataframe['high'], timeperiod=sell_params['ema_sell_size'])
        dataframe['emal_sell'] = ta.EMA(dataframe['low'], timeperiod=sell_params['ema_sell_size'])
        dataframe['emac_sell'] = ta.EMA(dataframe['close'], timeperiod=sell_params['ema_sell_size'])
        dataframe['emao_sell'] = ta.EMA(dataframe['open'], timeperiod=sell_params['ema_sell_size'])

        # BOLLINGER
        bollinger_buy = qtpylib.bollinger_bands(dataframe['close'], window=buy_params['bb_buy_size'], stds=2)
        dataframe['bb_lowerband_buy'] = bollinger_buy['lower']
        dataframe['bb_upperband_buy'] = bollinger_buy['upper']

        bollinger_sell = qtpylib.bollinger_bands(dataframe['close'], window=sell_params['bb_sell_size'], stds=2)
        dataframe['bb_lowerband_sell'] = bollinger_sell['lower']
        dataframe['bb_upperband_sell'] = bollinger_sell['upper']

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
                    ((dataframe['slowk'] >= buy_params['slowk_buy_bottom']) & (dataframe['slowk'] <= buy_params['slowk_buy_up']))
                    &
                    ((dataframe['slowd'] >= buy_params['slowd_buy_bottom']) & (dataframe['slowd'] <= buy_params['slowd_buy_up']))
                )
                &
                (
                    (
                        ( #Original buy condition
                            (dataframe['macd_buy'] > dataframe['macd_buy'].shift(1))
                            &
                            (dataframe['macdsignal_buy'] > dataframe['macdsignal_buy'].shift(1))
                        )
                        &
                        ( #Original buy condition
                            (dataframe['close'] > dataframe['close'].shift(1))
                        )
                        &
                        ( #Original buy condition
                            (dataframe['emac_buy'] >= dataframe['emao_buy'])
                            |
                            (dataframe['open'] < dataframe['emal_buy'])
                        )
                    )
                    |
                    ( # V2 Added buy condition w/ Bollingers bands
                        (dataframe['close'] <= dataframe['bb_lowerband_buy'])
                        |
                        (dataframe['open'] <= dataframe['bb_lowerband_buy'])
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
                        (dataframe['slowk'] <= sell_params['slowk_sell']) & (dataframe['slowd'] <= sell_params['slowd_sell'])
                )
                &
                (
                    (
                        ( #Original buy condition
                            (dataframe['macd_sell'] < dataframe['macd_sell'].shift(1))
                            &
                            (dataframe['macdsignal_sell'] < dataframe['macdsignal_sell'].shift(1))
                        )
                        &
                        ( #Original buy condition
                            (dataframe['emac_sell'] < dataframe['emao_sell'])
                            |
                            (dataframe['open'] >= dataframe['emah_sell']) ##V3 addon based on SmoothScalp
                        )
                    )
                    |
                    ( # V2 Added buy condition w/ Bollingers bands
                        (dataframe['close'] >= dataframe['bb_upperband_sell'])
                        |
                        (dataframe['open'] >= dataframe['bb_upperband_sell'])
                    )
                )
            ),
            'sell'] = 1
        return dataframe
    