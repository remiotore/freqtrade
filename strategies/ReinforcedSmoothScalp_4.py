
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, DatetimeIndex, merge
import numpy  # noqa

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class ReinforcedSmoothScalp_4(IStrategy):
    """
        this strategy is based around the idea of generating a lot of potentatils buys and make tiny profits on each trade

        we recommend to have at least 60 parallel trades at any time to cover non avoidable losses
    """


    minimal_roi = {
        "0": 0.02
    }




    stoploss = -0.1


    ticker_interval = '1m'

    resample_factor = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                        (dataframe['open'] < dataframe['ema_low']) &
                        (dataframe['adx'] > 30) &
                        (dataframe['mfi'] < 30) &
                        (
                                (dataframe['fastk'] < 30) &
                                (dataframe['fastd'] < 30) &
                                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
                        ) &
                        (dataframe['resample_sma'] < dataframe['close'])
                )





            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (
                            (
                                (dataframe['open'] >= dataframe['ema_high'])

                            ) |
                            (
                                    (qtpylib.crossed_above(dataframe['fastk'], 70)) |
                                    (qtpylib.crossed_above(dataframe['fastd'], 70))

                            )
                    ) & (dataframe['cci'] > 100)
            )
            ,
            'sell'] = 1
        return dataframe
