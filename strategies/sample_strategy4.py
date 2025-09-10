
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class sample_strategy4(IStrategy):
    """
    author@: Gert Wohlgemuth
    idea:
    The concept is about combining several common indicators, with a heavily smoothing, while trying to detect
    a none completed peak shape.
    """


    minimal_roi = {
        "0": 0.10
    }



    stoploss = -0.05

    ticker_interval = '5m'

    resample_factor = 12

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = StrategyHelper.resample(dataframe, self.ticker_interval, self.resample_factor)



        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi_smooth'] = ta.EMA(dataframe, timeperiod=11, price='mfi')
        dataframe['cci_smooth'] = ta.EMA(dataframe, timeperiod=11, price='cci')
        dataframe['rsi_smooth'] = ta.EMA(dataframe, timeperiod=11, price='rsi')


        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']


        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=1.6)
        dataframe['entry_bb_lowerband'] = bollinger['lower']
        dataframe['entry_bb_upperband'] = bollinger['upper']
        dataframe['entry_bb_middleband'] = bollinger['mid']

        dataframe['bpercent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (
                dataframe['bb_upperband'] - dataframe['bb_lowerband']) * 100

        dataframe['bsharp'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / (
            dataframe['bb_middleband'])


        dataframe['bsharp_slow'] = ta.SMA(dataframe, price='bsharp', timeperiod=11)
        dataframe['bsharp_medium'] = ta.SMA(dataframe, price='bsharp', timeperiod=8)
        dataframe['bsharp_fast'] = ta.SMA(dataframe, price='bsharp', timeperiod=5)


        dataframe['mfi_rsi_cci_smooth'] = (dataframe['rsi_smooth'] * 1.125 + dataframe['mfi_smooth'] * 1.125 +
                                           dataframe[
                                               'cci_smooth']) / 3

        dataframe['mfi_rsi_cci_smooth'] = ta.TEMA(dataframe, timeperiod=21, price='mfi_rsi_cci_smooth')

        dataframe['candle_size'] = (dataframe['close'] - dataframe['open']) * (
                dataframe['close'] - dataframe['open']) / 2

        dataframe['average'] = (dataframe['close'] + dataframe['open'] + dataframe['high'] + dataframe['low']) / 4
        dataframe['sma_slow'] = ta.SMA(dataframe, timeperiod=200, price='close')
        dataframe['sma_medium'] = ta.SMA(dataframe, timeperiod=100, price='close')
        dataframe['sma_fast'] = ta.SMA(dataframe, timeperiod=50, price='close')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (





















                (



                    (
                            (dataframe['average'].shift(5) > dataframe['average'].shift(4))
                            & (dataframe['average'].shift(4) > dataframe['average'].shift(3))
                            & (dataframe['average'].shift(3) > dataframe['average'].shift(2))
                            & (dataframe['average'].shift(2) > dataframe['average'].shift(1))
                            & (dataframe['average'].shift(1) < dataframe['average'].shift(0))
                            & (dataframe['low'].shift(1) < dataframe['bb_middleband'])
                            & (dataframe['cci'].shift(1) < -100)
                            & (dataframe['rsi'].shift(1) < 30)

                    )
                    |

                    (
                            (dataframe['low'] < dataframe['bb_middleband'])
                            & (dataframe['cci'] < -200)
                            & (dataframe['rsi'] < 30)
                            & (dataframe['mfi'] < 30)
                    )

                    |



                    (
                            (dataframe['mfi'] < 10)
                            & (dataframe['cci'] < -150)
                            & (dataframe['rsi'] < dataframe['mfi'])
                    )

                )

                &

                (dataframe['close'] > dataframe)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (


                        (
                                (dataframe['mfi_rsi_cci_smooth'] > 100)
                                & (dataframe['mfi_rsi_cci_smooth'].shift(1) > dataframe['mfi_rsi_cci_smooth'])
                                & (dataframe['mfi_rsi_cci_smooth'].shift(2) < dataframe['mfi_rsi_cci_smooth'].shift(1))
                                & (dataframe['mfi_rsi_cci_smooth'].shift(3) < dataframe['mfi_rsi_cci_smooth'].shift(2))
                        )

                        |


                        (
                            StrategyHelper.eight_green_candles(dataframe)
                        )
                        |


                        (
                                (dataframe['cci'] > 200)
                                & (dataframe['rsi'] > 70)
                        )

                )

            ),
            'sell'] = 1
        return dataframe


class StrategyHelper:
    """
        simple helper class to predefine a couple of patterns for our
        strategy
    """

    @staticmethod
    def seven_green_candles(dataframe):
        """
            evaluates if we are having 7 green candles in a row
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] < dataframe['close']) &
                (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) < dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) < dataframe['close'].shift(4)) &
                (dataframe['open'].shift(5) < dataframe['close'].shift(5)) &
                (dataframe['open'].shift(6) < dataframe['close'].shift(6)) &
                (dataframe['open'].shift(7) < dataframe['close'].shift(7))
        )

    @staticmethod
    def eight_green_candles(dataframe):
        """
            evaluates if we are having 8 green candles in a row
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] < dataframe['close']) &
                (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) < dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) < dataframe['close'].shift(4)) &
                (dataframe['open'].shift(5) < dataframe['close'].shift(5)) &
                (dataframe['open'].shift(6) < dataframe['close'].shift(6)) &
                (dataframe['open'].shift(7) < dataframe['close'].shift(7)) &
                (dataframe['open'].shift(8) < dataframe['close'].shift(8))
        )

    @staticmethod
    def eight_red_candles(dataframe, shift=0):
        """
            evaluates if we are having 8 red candles in a row
        :param self:
        :param dataframe:
        :param shift: shift the pattern by n
        :return:
        """
        return (
                (dataframe['open'].shift(shift) > dataframe['close'].shift(shift)) &
                (dataframe['open'].shift(1 + shift) > dataframe['close'].shift(1 + shift)) &
                (dataframe['open'].shift(2 + shift) > dataframe['close'].shift(2 + shift)) &
                (dataframe['open'].shift(3 + shift) > dataframe['close'].shift(3 + shift)) &
                (dataframe['open'].shift(4 + shift) > dataframe['close'].shift(4 + shift)) &
                (dataframe['open'].shift(5 + shift) > dataframe['close'].shift(5 + shift)) &
                (dataframe['open'].shift(6 + shift) > dataframe['close'].shift(6 + shift)) &
                (dataframe['open'].shift(7 + shift) > dataframe['close'].shift(7 + shift)) &
                (dataframe['open'].shift(8 + shift) > dataframe['close'].shift(8 + shift))
        )

    @staticmethod
    def four_green_one_red_candle(dataframe):
        """
            evaluates if we are having a red candle and 4 previous green
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] > dataframe['close']) &
                (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) < dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) < dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) < dataframe['close'].shift(4))
        )

    @staticmethod
    def four_red_one_green_candle(dataframe):
        """
            evaluates if we are having a green candle and 4 previous red
        :param self:
        :param dataframe:
        :return:
        """
        return (
                (dataframe['open'] < dataframe['close']) &
                (dataframe['open'].shift(1) > dataframe['close'].shift(1)) &
                (dataframe['open'].shift(2) > dataframe['close'].shift(2)) &
                (dataframe['open'].shift(3) > dataframe['close'].shift(3)) &
                (dataframe['open'].shift(4) > dataframe['close'].shift(4))
        )


    @staticmethod
    def resample( dataframe, interval, factor):


        df = dataframe.copy()
        df = df.set_index(DatetimeIndex(df['date']))
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }
        df = df.resample(str(int(interval[:-1]) * factor) + 'min', plotoschow=ohlc_dict)

        df['resample_sma'] = ta.SMA(df, timeperiod=25, price='close')
        df = df.drop(columns=['open', 'high', 'low', 'close'])
        df = df.resample(interval[:-1] + 'min')
        df = df.interpolate(method='time')
        df['date'] = df.index
        df.index = range(len(df))
        dataframe = merge(dataframe, df, on='date', how='left')
        return dataframe