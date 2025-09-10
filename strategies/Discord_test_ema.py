# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta


class test_ema(IStrategy):

    minimal_roi = {
        "0": 1,
    }

    stoploss = -0.1

    timeframe = '5m'

    period = 200
    history = 1000

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        test_data = dataframe.copy()
        # last 1200 candles
        test_data = test_data.iloc[(test_data['close'].count()-(self.history + self.period)):]
        test_data['talib_macd'] = ta.MACD(test_data)['macd']
        test_data['talib_ema'] = ta.EMA(test_data, timeperiod=self.period)
        test_data['talib_sma'] = ta.SMA(test_data, timeperiod=self.period)

        test_data_1 = dataframe.copy()
        # last 1199 candles
        test_data_1 = test_data_1.iloc[(test_data_1['close'].count()-(self.history + self.period -1)):]
        test_data_1['talib_macd'] = ta.MACD(test_data_1)['macd']
        test_data_1['talib_ema'] = ta.EMA(test_data_1, timeperiod=self.period)
        test_data_1['talib_sma'] = ta.SMA(test_data_1, timeperiod=self.period)


        print ("Pair: {}".format(metadata['pair'].replace("/","-")))
        print ("test_data: last {} candles".format(self.history + self.period))
        print(test_data)
        print ("test_data_1: last {} candles".format(self.history + self.period - 1))
        print(test_data_1)


#        test_data.to_csv('user_data/output/test-data-{}.csv'.format(metadata['pair'].replace("/","-")))
#        test_data_1.to_csv('user_data/output/test-data-1-{}.csv'.format(metadata['pair'].replace("/","-")))

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

            ),
            'sell'] = 1
        return dataframe
