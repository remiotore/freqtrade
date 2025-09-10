from datetime import datetime
import numpy as np  # noqa
from pandas import DataFrame, Series
from technical.indicators import zema, VIDYA, vwma
from freqtrade.persistence.trade_model import Trade
from technical.indicators import fibonacci_retracements

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Pmax_01(IStrategy):

    INTERFACE_VERSION = 3

    can_short: bool = False


    minimal_roi = {
        '0': 100,  # This is 10000%, which basically disables ROI
    }    


    stoploss = -1

    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = False

    timeframe = '1d'

    process_only_new_candles = True

    startup_candle_count: int = 30

    AccumulationLength = 32
    DistributionLength = 35
    SpringLength = 10
    UpthrustLength = 20

    Period = 10
    Multiplier = 3        
    Length = 10
    Type = 9
    Source = 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['fib'] = fibonacci_retracements(dataframe)

        heikinashi = qtpylib.heikinashi(dataframe)


        dataframe['ZLEMA'] = zema(dataframe, period=self.Period)
        dataframe['pm'], dataframe['pmx'] = pmax(dataframe, period=self.Period,  multiplier=self.Multiplier, length=self.Length, MAtype=self.Type,  src=self.Source)

        return dataframe
    
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['ZLEMA'], dataframe['pm'])) &
                (dataframe['fib'] < 0.5)
            ),
            'enter_long'] = 1
    
        return dataframe
    
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), 'exit_long'] = 0

        return dataframe
    
    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:

        if current_profit > 0.2:
            return 0.05
        if current_profit > 0.1:
            return 0.04
        if current_profit > 0.06:
            return 0.03
        if current_profit > 0.03:
            return 0.01

        return -1

def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'










    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx