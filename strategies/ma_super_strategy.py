from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ma_super_strategy(IStrategy):

    stoploss = -0.05 # Fermeture de l'ordre si perte de 5%

    ticker_interval = '1m' # Nous jouons avec les chandeliers de 1 minutes

    minimal_roi = { "0": 0.015 }

    populate_indicators_count = 0
    populate_buy_trend_count = 0
    populate_sell_trend_count = 0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        MaSuperStrategy.populate_indicators_count += 1
        print("populate_indicators({}) with metadata : {}".format(MaSuperStrategy.populate_indicators_count, metadata))
        if (MaSuperStrategy.populate_indicators_count == 1):
            print("Sample dataframe")
            print(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)

        dataframe['bb_lowerband'] = bollinger['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        MaSuperStrategy.populate_buy_trend_count += 1
        print("populate_buy_trend({}) with metadata : {}".format(MaSuperStrategy.populate_buy_trend_count, metadata))
        if (MaSuperStrategy.populate_buy_trend_count == 1):
            print("Sample dataframe")
            print(dataframe)
        dataframe.loc[
            (

                (dataframe['rsi'] < 30) &

                (dataframe['close'] < dataframe['bb_lowerband'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        MaSuperStrategy.populate_sell_trend_count += 1
        print("populate_sell_trend({}) with metadata : {}".format(MaSuperStrategy.populate_sell_trend_count, metadata))
        if (MaSuperStrategy.populate_sell_trend_count == 1):
            print("Sample dataframe")
            print(dataframe)
        dataframe.loc[
            (

                (dataframe['rsi'] > 50)
            ),
            'sell'] = 1
        return dataframe
