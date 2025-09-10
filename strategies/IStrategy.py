


from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class IStrategy(IStrategy):



    minimal_roi = {




        "0": 0.1
    }

    stoploss = -0.07




    ticker_interval = '1h'
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
    dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
    dataframe['long'] = ta.SMA(dataframe, timeperiod=6)
    dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
    dataframe['bb_low'] = bollinger['lower']
    dataframe['bb_mid'] = bollinger['mid']
    dataframe['bb_upper'] = bollinger['upper']

    dataframe['bb_perc'] = (dataframe['close'] - dataframe['bb_low']) / (dataframe['bb_upper'] - dataframe['bb_low'])
    return dataframe
def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
                ((dataframe['adx'] > 25) &
                qtpylib.crossed_above(dataframe['short'], dataframe['long'])) |
                (
                    (dataframe['macd'] > 0) &
                    ((dataframe['macd'] > dataframe['macdsignal']) |
                     ((dataframe['ao'] > 0) &
                    (dataframe['ao'].shift() < 0))) |
                    (dataframe['bb_perc'] < 0.1)
                )
        ),
        'buy'] = 1
    return dataframe
def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
        (
            ((dataframe['adx'] < 25) &
                (qtpylib.crossed_above(dataframe['long'], dataframe['short']))) |
            (
                    (dataframe['macd'] < 0) &
                    ((dataframe['macd'] < dataframe['macdsignal']) |
                     ((dataframe['ao'] < 0) &
                      (dataframe['ao'].shift() > 0)))
                    (dataframe['close'] > dataframe['high'].rolling(60).max().shift())
            )
        ),
        'sell'] = 1
    return dataframe
