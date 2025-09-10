"""
this strat buy deep and sell up and it's not smart sorry
u or gain profit or just hodl(when drawdown)
i fixed some of false signal, thank EMA(200)
when long drawdown, some time u get only exit_profit_offset! Because get false sell signal bellow buy price.
but u have no stop-loss and sell only profit

Do Backtesting first
freqtrade backtesting -s SmartMoneyStrategy --timerange 20210601- -i 1h -p DOT/USDT

Lets plot:
freqtrade plot-dataframe -s SmartMoneyStrategy --timerange 20210601- -i 1h -p DOT/USDT --indicators1 ema_200 --indicators2 cmf mfi

Params hyper-optable, just use class SmartMoneyStrategyHyperopt
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy SmartMoneyStrategyHyperopt --spaces buy sell --timerange 20210601- --dry-run-wallet 160 --stake 12 -i 1h -e 1000
"""

import numpy
import talib.abstract as ta
from pandas import DataFrame
from technical.indicators import chaikin_money_flow
from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter)


class SmartMoneyStrategy(IStrategy):

    minimal_roi = {
        "0": 10
    }

    stoploss = -1

    timeframe = '1h'
    exit_profit_only = True
    exit_profit_offset = 0.01

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['cmf'] = chaikin_money_flow(dataframe, period=20)

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['ema_200']) &
                    (dataframe['mfi'] < 35) &
                    (dataframe['cmf'] < -0.07)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['ema_200']) &
                    (dataframe['mfi'] > 70) &
                    (dataframe['cmf'] > 0.20)
            ),
            'sell'] = 1

        return dataframe

