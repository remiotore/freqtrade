# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import DecimalParameter, IntParameter

def line2arr(line, size=-1):
    if size <= 0:
        return np.array(line.array)
    else:
        return np.array(line.get(size=size))

def na(val):
    return val != val

def nz(x, y=None):
    if isinstance(x, np.generic):
        return x.fillna(y or 0)
    if x != x:
        if y is not None:
            return y
        return 0
    return x

def barssince(condition, occurrence=0):
    cond_len = len(condition)
    occ = 0
    since = 0
    res = float('nan')
    while cond_len - (since+1) >= 0:
        print(since)
        cond = condition[cond_len-(since+1)]
        print(cond)
        if cond and not cond != cond:
            if occ == occurrence:
                res = since
                break
            occ += 1
        since += 1
    return res

def valuewhen(condition, source, occurrence=0):
    res = float('nan')
    since = barssince(condition, occurrence)
    print(since)
    if since is not None:
        res = source[-(since+1)]
    return res


class RSIDivTirail(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "candle_ratio": 0.062,
        "pin_ratio": 4,
        "rsi_high": 70,
        "rsi_low": 18,
        "top_pin_ratio": 0.029,
    }
    # Sell hyperspace params:
    sell_params = {
    }

    #pin_ratio = IntParameter(2, 10, default=buy_params['pin_ratio'], space='buy', optimize=True)
    #top_pin_ratio = DecimalParameter(0, 1, default=buy_params['top_pin_ratio'], space='buy', optimize=True)
    #candle_ratio = DecimalParameter(0.01, 0.12, default=buy_params['candle_ratio'], space='buy', optimize=True)
    #rsi_low = IntParameter(10, 100,
    #                           default=buy_params['rsi_low'], space='buy', optimize=True)
    #rsi_high = IntParameter(10, 100,
    #                           default=buy_params['rsi_high'], space='buy', optimize=True)
    #stoch_low = IntParameter(10, 50,
    #                           default=buy_params['stoch_low'], space='buy', optimize=True)
    #stoch_high = IntParameter(50, 95,
    #                           default=buy_params['stoch_high'], space='buy', optimize=True)
    #cci_low = IntParameter(-300, -50,
    #                           default=buy_params['cci_low'], space='buy', optimize=True)
    #cci_high = IntParameter(50, 300,
    #                           default=buy_params['cci_high'], space='buy', optimize=True)

    # ROI table:
    minimal_roi = {
        "0": 0.05,
    }

    # Stoploss:
    stoploss = -0.1

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_custom_stoploss = True

    rangeUpper = 60
    rangeLower = 5

    def in_range(self, condition):
        """
        _inRange(cond) =>
            bars = barssince(cond == true)
            rangeLower <= bars and bars <= rangeUpper
        """
        bars = 0
        while True:
            if not condition.shift(bars):
                break
            bars += 1
        return (self.rangeLower <= bars and bars <= self.rangeUpper)


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        study(title="Divergence Indicator", format=format.price, resolution="")
        len = input(title="RSI Period", minval=1, defval=14)
        src = input(title="RSI Source", defval=close)
        lbR = input(title="Pivot Lookback Right", defval=5)
        lbL = input(title="Pivot Lookback Left", defval=5)
        rangeUpper = input(title="Max of Lookback Range", defval=60)
        rangeLower = input(title="Min of Lookback Range", defval=5)
        plotBull = input(title="Plot Bullish", defval=true)
        plotHiddenBull = input(title="Plot Hidden Bullish", defval=false)
        plotBear = input(title="Plot Bearish", defval=true)
        plotHiddenBear = input(title="Plot Hidden Bearish", defval=false)
        bearColor = color.red
        bullColor = color.green
        hiddenBullColor = color.new(color.green, 80)
        hiddenBearColor = color.new(color.red, 80)
        textColor = color.white
        noneColor = color.new(color.white, 100)
        osc = rsi(src, len)
        """

        len = 14
        src = dataframe['close']
        lbL = 5
        plotBull = True
        plotHiddenBull = False
        plotBear = True
        plotHiddenBear = False
        dataframe['osc'] = ta.RSI(src, len)

        #plFound = na(pivotlow(osc, lbL, lbR)) ? false : true
        dataframe['min'] = dataframe['close'].rolling(lbL).min()
        dataframe['prevMin'] = np.where(dataframe['min'] > dataframe['min'].shift(), dataframe['min'].shift(), dataframe['min'])
        dataframe.loc[
            (dataframe['min'] != dataframe['prevMin'])
        , 'plFound'] = 1
        dataframe['plFound'] = dataframe['plFound'].fillna(0)

        # phFound = na(pivothigh(osc, lbL, lbR)) ? false : true
        dataframe['max'] = dataframe['close'].rolling(lbL).max()
        dataframe['prevMax'] = np.where(dataframe['max'] < dataframe['max'].shift(), dataframe['max'].shift(), dataframe['max'])
        dataframe.loc[
            (dataframe['max'] != dataframe['prevMax'])
        , 'phFound'] = 1
        dataframe['phFound'] = dataframe['phFound'].fillna(0)


        #------------------------------------------------------------------------------
        # Regular Bullish
        # Osc: Higher Low
        # oscHL = osc[lbR] > valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        dataframe.loc[
            (
                (dataframe['osc'] > valuewhen(dataframe['plFound'], dataframe['osc'], 1)) &
                (self.in_range(dataframe['plFound'].shift(1)))
             )
        , 'oscHL'] = 1

        # Price: Lower Low
        # priceLL = low[lbR] < valuewhen(plFound, low[lbR], 1)
        dataframe.loc[
            (dataframe['low'] < valuewhen(dataframe['plFound'], dataframe['low'], 1))
            , 'priceLL'] = 1
        #bullCond = plotBull and priceLL and oscHL and plFound
        dataframe.loc[
            (
                (dataframe['priceLL'] == 1) &
                (dataframe['oscHL'] == 1) &
                (dataframe['plFound'] == 1)
            )
            , 'bullCond'] = 1

        # plot(
        #      plFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bullish",
        #      linewidth=2,
        #      color=(bullCond ? bullColor : noneColor)
        #      )
        #
        # plotshape(
        #      bullCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bullish Label",
        #      text=" Bull ",
        #      style=shape.labelup,
        #      location=location.absolute,
        #      color=bullColor,
        #      textcolor=textColor
        #      )

        # //------------------------------------------------------------------------------
        # // Hidden Bullish
        # // Osc: Lower Low
        #
        # oscLL = osc[lbR] < valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
        #
        # // Price: Higher Low
        #
        # priceHL = low[lbR] > valuewhen(plFound, low[lbR], 1)
        # hiddenBullCond = plotHiddenBull and priceHL and oscLL and plFound
        #
        # plot(
        #      plFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bullish",
        #      linewidth=2,
        #      color=(hiddenBullCond ? hiddenBullColor : noneColor)
        #      )
        #
        # plotshape(
        #      hiddenBullCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bullish Label",
        #      text=" H Bull ",
        #      style=shape.labelup,
        #      location=location.absolute,
        #      color=bullColor,
        #      textcolor=textColor
        #      )
        #
        # //------------------------------------------------------------------------------
        # // Regular Bearish
        # // Osc: Lower High
        #
        # oscLH = osc[lbR] < valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        #
        # // Price: Higher High
        #
        # priceHH = high[lbR] > valuewhen(phFound, high[lbR], 1)
        #
        # bearCond = plotBear and priceHH and oscLH and phFound
        #
        # plot(
        #      phFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bearish",
        #      linewidth=2,
        #      color=(bearCond ? bearColor : noneColor)
        #      )
        #
        # plotshape(
        #      bearCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Regular Bearish Label",
        #      text=" Bear ",
        #      style=shape.labeldown,
        #      location=location.absolute,
        #      color=bearColor,
        #      textcolor=textColor
        #      )
        #
        # //------------------------------------------------------------------------------
        # // Hidden Bearish
        # // Osc: Higher High
        #
        # oscHH = osc[lbR] > valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
        #
        # // Price: Lower High
        #
        # priceLH = high[lbR] < valuewhen(phFound, high[lbR], 1)
        #
        # hiddenBearCond = plotHiddenBear and priceLH and oscHH and phFound
        #
        # plot(
        #      phFound ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bearish",
        #      linewidth=2,
        #      color=(hiddenBearCond ? hiddenBearColor : noneColor)
        #      )
        #
        # plotshape(
        #      hiddenBearCond ? osc[lbR] : na,
        #      offset=-lbR,
        #      title="Hidden Bearish Label",
        #      text=" H Bear ",
        #      style=shape.labeldown,
        #      location=location.absolute,
        #      color=bearColor,
        #      textcolor=textColor
        #  )"""



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['bullCond'] > 0) &
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.to_csv('user_data/csvs/%s_%s.csv' % (self.__class__.__name__, metadata["pair"].replace("/", "_")))
        dataframe.loc[
            (
                (
                        (dataframe['volume'] < 0)
                 )
            ),
            'sell'] = 1
        return dataframe
