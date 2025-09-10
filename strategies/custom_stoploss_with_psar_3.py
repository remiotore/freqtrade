


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.state import RunMode

class custom_stoploss_with_psar_3(IStrategy):
    """
    this is an example class, implementing a PSAR based trailing stop loss
    you are supposed to take the `custom_stoploss()` and `populate_indicators()`
    parts and adapt it to your own strategy

    the populate_buy_trend() function is pretty nonsencial
    """

    custom_info = {}
    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        result = 1
        if self.custom_info and pair in self.custom_info and trade:


            relative_sl = None
            if self.dp:

                if self.dp.runmode.value in ('backtest', 'hyperopt'):
                    relative_sl = self.custom_info[pair].loc[current_time]['sar']


                else:

                    dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                             timeframe=self.timeframe)


                    relative_sl = dataframe['sar'].iat[-1]

            if (relative_sl is not None):


                new_stoploss = (current_rate-relative_sl)/current_rate

                result = new_stoploss - 1

        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sar'] = ta.SAR(dataframe)
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_info[metadata['pair']] = dataframe[['date', 'sar']].copy().set_index('date')



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: buys when SAR is smaller then candle before
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['sar'] < dataframe['sar'].shift())
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Placeholder Strategy: does nothing
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[:, 'sell'] = 0
        return dataframe
