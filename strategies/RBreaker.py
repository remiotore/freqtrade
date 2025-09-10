from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy

import talib.abstract as ta

from freqtrade.strategy.interface import IStrategy

import datetime

import pandas as pd

class RBreakerStrategy(IStrategy):
    INTERFACE_VERSION: int = 3
    # ROI table:
    minimal_roi = {"0": 0.15, "30": 0.1, "60": 0.05}
    # minimal_roi = {"0": 1}

    # Stoploss:
    stoploss = -0.265

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    timeframe = "5m"

 #   can_short = True

    setup_coef = 0.35
    break_coef = 0.25
    enter_coef1 = 1.07
    enter_coef2 = 0.07
    fixed_size = 0.03

    start_time=0
    end_time=0
    Ssetup=0
    Bsetup=0
    Senter=0
    Benter=0
    Bbreak=0
    Sbreak=0
    TradeTime = datetime.time(hour=23, minute=50, second=0)




    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['TradeTime']=dataframe['date'].dt.time


        self.end_time = dataframe.iloc[-1]['date']
    #    self.end_time = dataframe['date']


        self.start_time = self.end_time + datetime.timedelta(days=-1)
        self.start_time = self.start_time.strftime("%Y%m%d")
        self.end_time = self.end_time.strftime("%Y%m%d")

        df2 = dataframe[(dataframe['date'] >= self.start_time) & (dataframe['date'] < self.end_time)]

        DayHigh = df2.max()['high']
        DayLow = df2.min()['low']
        DayClose = df2.iloc[-1]['close']



        self.Ssetup = DayHigh + self.setup_coef * (DayClose - DayLow)  # watch sell
        self.Bsetup = DayLow - self.setup_coef * (DayHigh - DayClose)  # watch buy

        self.Senter = self.enter_coef1 / 2 * (DayHigh + DayLow) - self.enter_coef2 * DayLow
        self.Benter = self.enter_coef1 / 2 * (DayHigh + DayLow) - self.enter_coef2 * DayHigh

        self.Bbreak = self.Ssetup + self.break_coef * (self.Ssetup - self.Bsetup)
        self.Sbreak = self.Bsetup - self.break_coef * (self.Ssetup - self.Bsetup)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
           # Add your trend following buy signals here
            dataframe.loc[
                ((dataframe['close'] > self.Bbreak) &
                (dataframe['close'].shift(1) < self.Bbreak))|
                ((dataframe['close'] > self.Benter) &
                (dataframe['close'].shift(1) < self.Benter) &
                (dataframe[(dataframe['date'] >= self.end_time)].min()['high']< self.Bsetup) &
                ((self.Senter-self.Benter) /self.Benter > self.fixed_size)),
                'enter_long'] = 1

            # Add your trend following sell signals here
            dataframe.loc[
                ((dataframe['high'] < self.Sbreak) &
                (dataframe['close'].shift(1) > self.Sbreak))|
                ((dataframe['close'] < self.Senter) &
                (dataframe['close'].shift(1) > self.Senter) &
                (dataframe[(dataframe['date'] >= self.end_time)].max()['high'] > self.Ssetup) &
                ((self.Senter - self.Benter) / self.Benter > self.fixed_size)),
                'enter_short'] = 1
 #           print(dataframe)
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

            # Add your trend following exit signals for long positions here
            dataframe.loc[
                ((dataframe['high'] < self.Sbreak) &
                 (dataframe['close'].shift(1) > self.Sbreak)) |
                ((dataframe['close'] < self.Senter) &
                 (dataframe['close'].shift(1) > self.Senter) &
                 (dataframe[(dataframe['date'] >= self.end_time)].max()['high'] > self.Ssetup))|
                (dataframe['TradeTime'] > self.TradeTime),
                'exit_long'] = 1

            # Add your trend following exit signals for short positions here
            dataframe.loc[
                ((dataframe['close'] > self.Bbreak) &
                 (dataframe['close'].shift(1) < self.Bbreak)) |
                ((dataframe['close'] > self.Benter) &
                 (dataframe['close'].shift(1) < self.Benter) &
                 (dataframe[(dataframe['date'] >= self.end_time)].min()['high'] < self.Bsetup))|
                (dataframe['TradeTime']> self.TradeTime),
                'exit_short'] = 1
     #       print(dataframe)
            return dataframe

