
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame



class CofiBitStrategy_3(IStrategy):
    """
        taken from slack by user CofiBit
    """


    minimal_roi = {
        "40": 0.05,
        "30": 0.06,
        "20": 0.07,
        "0": 0.10
    }


    stoploss = -0.25

    timeframe = '15m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        dataframe['adx'] = ta.ADX(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['open'] < dataframe['ema_low']) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &

                (dataframe['fastk'] < 30) &
                (dataframe['fastd'] < 30) &
                (dataframe['adx'] > 30)
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
                (dataframe['open'] >= dataframe['ema_high'])
            ) |
            (


                    (qtpylib.crossed_above(dataframe['fastk'], 70)) |
                    (qtpylib.crossed_above(dataframe['fastd'], 70))
            ),
            'sell'] = 1

        return dataframe
