


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series, DatetimeIndex, merge

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class divergences_test(IStrategy):
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

    trailing_stop_positive = 0.05
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

        dataframe['mean24volume'] = dataframe.volume.rolling(24).mean() 

        dataframe['mean68close'] = dataframe.close.rolling(68).mean() 





        dataframe['bullish_div'] = (
                                        ( dataframe['close'].shift(4) > dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(3) > dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(2) < dataframe['close'].shift(1) ) & 
                                        ( dataframe['close'].shift(2) < dataframe['close'] )
                                   ) 





        dataframe['bearish_div'] = (
                                        ( dataframe['close'].shift(4) < dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(3) < dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(2) > dataframe['close'].shift(1) ) & 
                                        ( dataframe['close'].shift(2) > dataframe['close'] )
                                    )

        dataframe['cci_one'] = ta.CCI(dataframe, timeperiod=170)
        dataframe['cci_two'] = ta.CCI(dataframe, timeperiod=34)



        dataframe['adx'] = ta.ADX(dataframe)




























        dataframe['cci'] = ta.CCI(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)










        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']







        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['mfi'] = ta.MFI(dataframe)





        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
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
















        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)








        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)



        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']




















































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
                        (dataframe['rsi'] <= 40) &
                        (dataframe['bullish_div'])
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
                    (dataframe['bearish_div'])
                ),
                'sell'] = 1

            return dataframe