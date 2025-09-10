

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy


class default_strategy_470(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    minimal_roi = {
        "40":  0.0,         #in 40min
        "30":  0.01,        #in 30min
        "20":  0.02,        #in 20min
        "0":  0.04
    }

    stoploss = -0.10

    ticker_interval = 5

    slippage = 0.01
    
    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """






        """

        dataframe['cci'] = ta.CCI(dataframe)
        """















        """

        dataframe['roc'] = ta.ROC(dataframe)
        """

        dataframe['rsi'] = ta.RSI(dataframe)












        """

        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']
        """
































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

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """













        dataframe.loc[ (dataframe['rsi'] < 35), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """















        dataframe.loc[ (qtpylib.crossed_above(dataframe['rsi'], 70)), 'sell'] = 1
        return dataframe
