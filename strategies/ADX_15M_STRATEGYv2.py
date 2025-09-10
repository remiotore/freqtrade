
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class ADX_15M_STRATEGYv2(IStrategy):
    ticker_interval = '15m'

    minimal_roi = {
        "0": 0.10313,
        "102": 0.07627,
        "275": 0.04228,
        "588": 0
    }

    stoploss = -0.31941

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=14)

        dataframe['sell-adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['sell-plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)
        dataframe['sell-minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)
        dataframe['sell-sar'] = ta.SAR(dataframe)
        dataframe['sell-mom'] = ta.MOM(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (



                    (qtpylib.crossed_above(dataframe['minus_di'], dataframe['plus_di']))

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > 91) &

                    (dataframe['sell-minus_di'] > 91) &

                    (qtpylib.crossed_above(dataframe['sell-plus_di'], dataframe['sell-minus_di']))

            ),
            'sell'] = 1
        return dataframe

