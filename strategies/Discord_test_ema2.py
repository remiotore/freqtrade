# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta


class test_ema2(IStrategy):

    minimal_roi = {
        "0": 1,
    }

    stoploss = -0.1

    timeframe = '5m'

#    ma_period = 50
#    history_1 = 50
    ma_period = history_1 = 50
    history_2 = 1000

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        test_data_1 = dataframe.copy()
        test_data_1 = test_data_1.iloc[(test_data_1['close'].count()-(self.history_1 + self.ma_period)):]
        test_data_1['macdhist'] = ta.MACD(test_data_1)['macdhist']
        test_data_1['ema'] = ta.EMA(test_data_1, timeperiod=self.ma_period)
        test_data_1['sma'] = ta.SMA(test_data_1, timeperiod=self.ma_period)

        test_data_2 = dataframe.copy()
        test_data_2 = test_data_2.iloc[(test_data_2['close'].count()-(self.history_2 + self.ma_period)):]
        test_data_2['macdhist'] = ta.MACD(test_data_2)['macdhist']
        test_data_2['ema'] = ta.EMA(test_data_2, timeperiod=self.ma_period)
        test_data_2['sma'] = ta.SMA(test_data_2, timeperiod=self.ma_period)

        print ("Pair: {}".format(metadata['pair'].replace("/","-")))
        print ("test_data_1: {} candles".format(test_data_1['close'].count()))
        print(test_data_1)
        print ("test_data_2: {} candles".format(test_data_2['close'].count()))
        print(test_data_2)

#        test_data_1.to_csv('user_data/output/test-data-1-{}.csv'.format(metadata['pair'].replace("/","-")))
#        test_data_2.to_csv('user_data/output/test-data-2-{}.csv'.format(metadata['pair'].replace("/","-")))

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
