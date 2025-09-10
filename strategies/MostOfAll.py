
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import numpy as np
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.hyper import DecimalParameter
from freqtrade.persistence import Trade
from datetime import datetime
from freqtrade.strategy import stoploss_from_open


def MOST(dataframe, length=8, percent=2, MAtype=1):
    """Partial implementation of MOST indicator."""

    data = dataframe.copy()

    if MAtype==1:
        data['exma']=ta.EMA(data, timeperiod = length)
    elif MAtype==2:
        data['exma']=ta.DEMA(data, timeperiod = length)
    elif MAtype==3:
        data['exma']=ta.T3(data, timeperiod = length)

    data['basic_ub'] = data['exma'] * (1+percent/100)
    data['basic_lb'] = data['exma'] * (1-percent/100)

    data['final_ub'] = 0.00
    data['final_lb'] = 0.00
    for i in range(length, len(data)):
        data['final_ub'].iat[i] = data['basic_ub'].iat[i] if data['basic_ub'].iat[i] < data['final_ub'].iat[i - 1] or data['exma'].iat[i - 1] > data['final_ub'].iat[i - 1] else data['final_ub'].iat[i - 1]
        data['final_lb'].iat[i] = data['basic_lb'].iat[i] if data['basic_lb'].iat[i] > data['final_lb'].iat[i - 1] or data['exma'].iat[i - 1] < data['final_lb'].iat[i - 1] else data['final_lb'].iat[i - 1]

    data['most'] = 0.00
    for i in range(length, len(data)):
        data['most'].iat[i] = data['final_ub'].iat[i] if data['most'].iat[i - 1] == data['final_ub'].iat[i - 1] and data['exma'].iat[i] <= data['final_ub'].iat[i] else \
                        data['final_lb'].iat[i] if data['most'].iat[i - 1] == data['final_ub'].iat[i - 1] and data['exma'].iat[i] >  data['final_ub'].iat[i] else \
                        data['final_lb'].iat[i] if data['most'].iat[i - 1] == data['final_lb'].iat[i - 1] and data['exma'].iat[i] >= data['final_lb'].iat[i] else \
                        data['final_ub'].iat[i] if data['most'].iat[i - 1] == data['final_lb'].iat[i - 1] and data['exma'].iat[i] <  data['final_lb'].iat[i] else 0.00

    data['trend'] = np.where((data['most'] > 0.00), np.where((data['exma'] < data['most']), 0, 1), np.NaN)

    data.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    data.fillna(0, inplace=True)

    return data


class MostOfAll(IStrategy):
    """
        My second humble strategy using a MOST alike indicator
        Changelog:
            0.9 Initial version, improvements needed

        https://github.com/cyberjunky/freqtrade-strategies
        https://www.tradingview.com/scripts/most/

    """


    stoploss = -0.2

    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    use_custom_stoploss = True

    timeframe = '5m'


    minimal_roi = {
        "0": 0.08,
        "36": 0.031,
        "50": 0.021,
        "60": 0.01,
        "70": 0
    }

    @property
    def plot_config(self):
        """Buildin plot config."""
        return {

            'main_plot': {
                    'most': {'color': 'darkpurple'},
                    'exma': {'color': 'green'}
            },
            'subplots': {

                "trend": {
                    'trend': {'color': 'blue'}
                }
            }
        }

    pHSL = DecimalParameter(-0.500, -0.040, default=-0.99, decimals=3, space='sell', load=True)

    pPF_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.009, decimals=3, space='sell', load=True)

    pPF_2 = DecimalParameter(0.040, 0.100, default=0.040, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.020, decimals=3, space='sell', load=True)

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Custom stoploss calculation with thresholds and based on linear curve."""

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value




        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        most_df = MOST(dataframe, length=14)
        dataframe['most'] = most_df['most']
        dataframe['exma'] = most_df['exma']
        dataframe['trend'] = most_df['trend']

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        If the bullish fractal is active and below the teeth of the gator -> buy
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        conditions = []
        conditions.append(
            (
                (qtpylib.crossed_above(dataframe['most'], dataframe['exma'])) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy']=1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        If the bearish fractal is active and above the teeth of the gator -> sell
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        conditions = []
        conditions.append(
            (
                (qtpylib.crossed_above(dataframe['exma'], dataframe['most'])) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell']=1

        return dataframe
