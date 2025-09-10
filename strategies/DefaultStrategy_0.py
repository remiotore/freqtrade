

"""
Created on Thu Aug 27 21:57:48 2020

@author: alex
"""


import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy


class DefaultStrategy_0(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    minimal_roi = {
        "40": 0.0,
        "30": 0.01,
        "20": 0.02,
        "0": 0.04
    }

    stoploss = -0.10

    ticker_interval = '5m'

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        """

        dataframe['cci'] = ta.CCI(dataframe)
        """

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        """

        dataframe['roc'] = ta.ROC(dataframe)
        """

        dataframe['rsi'] = ta.RSI(dataframe)

        dataframe['fisher_rsi'] = fishers_inverse(dataframe['rsi'])

        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        """

        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']
        """






        dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)



        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']


        """

        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)

        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)

        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]

        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]

        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]
        """


        """

        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)

        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)

        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)

        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)

        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)

        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)
        """


        """

        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)

        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]

        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]

        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]

        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]

        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]
        """



        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['rsi'] < 35) &
                (dataframe['fastd'] < 35) &
                (dataframe['adx'] > 30) &
                (dataframe['plus_di'] > 0.5)
            ) |
            (
                (dataframe['adx'] > 65) &
                (dataframe['plus_di'] > 0.5)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (
                    (qtpylib.crossed_above(dataframe['rsi'], 70)) |
                    (qtpylib.crossed_above(dataframe['fastd'], 70))
                ) &
                (dataframe['adx'] > 10) &
                (dataframe['minus_di'] > 0)
            ) |
            (
                (dataframe['adx'] > 70) &
                (dataframe['minus_di'] > 0.5)
            ),
            'sell'] = 1
        return dataframe