"""
Supertrend strategy:
* Description: Generate a 3 supertrend indicators for 'buy' strategies & 3 supertrend indicators for 'sell' strategies
               Buys if the 3 'buy' indicators are 'up'
               Sells if the 3 'sell' indicators are 'down'
* Author: @juankysoriano (Juan Carlos Soriano)
* github: https://github.com/juankysoriano/
*** NOTE: This Supertrend strategy is just one of many possible strategies using `Supertrend` as indicator. It should on any case used at your own risk.
          It comes with at least a couple of caveats:
            1. The implementation for the `supertrend` indicator is based on the following discussion: https://github.com/freqtrade/freqtrade-strategies/issues/30 . Concretelly https://github.com/freqtrade/freqtrade-strategies/issues/30#issuecomment-853042401
            2. The implementation for `supertrend` on this strategy is not validated; meaning this that is not proven to match the results by the paper where it was originally introduced or any other trusted academic resources
"""

import logging
from numpy.lib import math
from freqtrade.strategy import IStrategy, IntParameter, BooleanParameter, DecimalParameter
from freqtrade.strategy import informative, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import warnings
import pandas as pd
from pandas_ta import supertrend as pandasupertrend, rma as pandarma
from typing import Optional, Any, Callable, Dict, List
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal, Real
import freqtrade.vendor.qtpylib.indicators as qtpylib
from numpy import nan as npNaN

import pandas_ta as pta
from freqtrade.persistence import Trade
from functools import reduce





from datetime import datetime, timedelta


def TA_get_min_max(series1: pd.Series, series2: pd.Series, function: str = "min"):
    """Find min or max value between two lists for each index"""
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)


class TAIndicatorMixin:
    """Util mixin indicator class"""

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        """Check if fillna flag is True.

        Args:
            series(pandas.Series): calculated indicator series.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.

        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(method='bfill')
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range


class TAAccDistIndexIndicator(TAIndicatorMixin):
    """Accumulation/Distribution Index (ADI)

    Acting as leading indicator of price movements.

    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        clv = ((self._close - self._low) - (self._high - self._close)) / (
            self._high - self._low
        )
        clv = clv.fillna(0.0)  # float division by zero
        adi = clv * self._volume
        self._adi = adi.cumsum()

    def acc_dist_index(self) -> pd.Series:
        """Accumulation/Distribution Index (ADI)

        Returns:
            pandas.Series: New feature generated.
        """
        adi = self._check_fillna(self._adi, value=0)
        return pd.Series(adi, name="adi")


class TAOnBalanceVolumeIndicator(TAIndicatorMixin):
    """On-balance volume (OBV)

    It relates price and volume in the stock market. OBV is based on a
    cumulative total volume.

    https://en.wikipedia.org/wiki/On-balance_volume

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        obv = np.where(self._close < self._close.shift(1), -self._volume, self._volume)
        self._obv = pd.Series(obv, index=self._close.index).cumsum()

    def on_balance_volume(self) -> pd.Series:
        """On-balance volume (OBV)

        Returns:
            pandas.Series: New feature generated.
        """
        obv = self._check_fillna(self._obv, value=0)
        return pd.Series(obv, name="obv")


class TAWilliamsRIndicator(TAIndicatorMixin):
    """Williams %R

    Developed by Larry Williams, Williams %R is a momentum indicator that is
    the inverse of the Fast Stochastic Oscillator. Also referred to as %R,
    Williams %R reflects the level of the close relative to the highest high
    for the look-back period. In contrast, the Stochastic Oscillator reflects
    the level of the close relative to the lowest low. %R corrects for the
    inversion by multiplying the raw value by -100. As a result, the Fast
    Stochastic Oscillator and Williams %R produce the exact same lines, only
    the scaling is different. Williams %R oscillates from 0 to -100.

    Readings from 0 to -20 are considered overbought. Readings from -80 to -100
    are considered oversold.

    Unsurprisingly, signals derived from the Stochastic Oscillator are also
    applicable to Williams %R.

    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.

    https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r

    The Williams %R oscillates from 0 to -100. When the indicator produces
    readings from 0 to -20, this indicates overbought market conditions. When
    readings are -80 to -100, it indicates oversold market conditions.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period.
        fillna(bool): if True, fill nan values with -50.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lbp: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._lbp = lbp
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._lbp
        highest_high = self._high.rolling(
            self._lbp, min_periods=min_periods
        ).max()  # highest high over lookback period lbp
        lowest_low = self._low.rolling(
            self._lbp, min_periods=min_periods
        ).min()  # lowest low over lookback period lbp
        self._wr = -100 * (highest_high - self._close) / (highest_high - lowest_low)

    def williams_r(self) -> pd.Series:
        """Williams %R

        Returns:
            pandas.Series: New feature generated.
        """
        wr_series = self._check_fillna(self._wr, value=-50)
        return pd.Series(wr_series, name="wr")


class TACumulativeReturnIndicator(TAIndicatorMixin):
    """Cumulative Return (CR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._cr = (self._close / self._close.iloc[0]) - 1
        self._cr *= 100

    def cumulative_return(self) -> pd.Series:
        """Cumulative Return (CR)

        Returns:
            pandas.Series: New feature generated.
        """
        cum_ret = self._check_fillna(self._cr, value=-1)
        return pd.Series(cum_ret, name="cum_ret")


class TAMFIIndicator(TAIndicatorMixin):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        typical_price = (self._high + self._low + self._close) / 3.0
        up_down = np.where(
            typical_price > typical_price.shift(1),
            1,
            np.where(typical_price < typical_price.shift(1), -1, 0),
        )
        mfr = typical_price * self._volume * up_down

        min_periods = 0 if self._fillna else self._window
        n_positive_mf = mfr.rolling(self._window, min_periods=min_periods).apply(
            lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), raw=True
        )
        n_negative_mf = abs(
            mfr.rolling(self._window, min_periods=min_periods).apply(
                lambda x: np.sum(np.where(x < 0.0, x, 0.0)), raw=True
            )
        )



        mfi = n_positive_mf / n_negative_mf
        self._mfi = 100 - (100 / (1 + mfi))

    def money_flow_index(self) -> pd.Series:
        """Money Flow Index (MFI)

        Returns:
            pandas.Series: New feature generated.
        """
        mfi = self._check_fillna(self._mfi, value=50)
        return pd.Series(mfi, name=f"mfi_{self._window}")


class TA_AroonIndicator(TAIndicatorMixin):
    """Aroon Indicator

    Identify when trends are likely to change direction.

    Aroon Up = ((N - Days Since N-day High) / N) x 100
    Aroon Down = ((N - Days Since N-day Low) / N) x 100
    Aroon Indicator = Aroon Up - Aroon Down

    https://www.investopedia.com/terms/a/aroon.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 25, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        rolling_close = self._close.rolling(
            self._window, min_periods=min_periods)
        self._aroon_up = rolling_close.apply(
            lambda x: float(np.argmax(x) + 1) / self._window * 100, raw=True
        )
        self._aroon_down = rolling_close.apply(
            lambda x: float(np.argmin(x) + 1) / self._window * 100, raw=True
        )

    def aroon_up(self) -> pd.Series:
        """Aroon Up Channel

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_up_series = self._check_fillna(self._aroon_up, value=0)
        return pd.Series(aroon_up_series, name=f"aroon_up_{self._window}")

    def aroon_down(self) -> pd.Series:
        """Aroon Down Channel

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_down_series = self._check_fillna(self._aroon_down, value=0)
        return pd.Series(aroon_down_series, name=f"aroon_down_{self._window}")

    def aroon_indicator(self) -> pd.Series:
        """Aroon Indicator

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_diff = self._aroon_up - self._aroon_down
        aroon_diff = self._check_fillna(aroon_diff, value=0)
        return pd.Series(aroon_diff, name=f"aroon_ind_{self._window}")


class TAADXIndicator(TAIndicatorMixin):
    """Average Directional Movement Index (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        if self._window == 0:
            raise ValueError("window may not be 0")

        close_shift = self._close.shift(1)
        pdm = TA_get_min_max(self._high, close_shift, "max")
        pdn = TA_get_min_max(self._low, close_shift, "min")
        diff_directional_movement = pdm - pdn

        self._trs_initial = np.zeros(self._window - 1)
        self._trs = np.zeros(len(self._close) - (self._window - 1))
        self._trs[0] = diff_directional_movement.dropna()[
            0: self._window].sum()
        diff_directional_movement = diff_directional_movement.reset_index(
            drop=True)

        for i in range(1, len(self._trs) - 1):
            self._trs[i] = (
                self._trs[i - 1]
                - (self._trs[i - 1] / float(self._window))
                + diff_directional_movement[self._window + i]
            )

        diff_up = self._high - self._high.shift(1)
        diff_down = self._low.shift(1) - self._low
        pos = abs(((diff_up > diff_down) & (diff_up > 0)) * diff_up)
        neg = abs(((diff_down > diff_up) & (diff_down > 0)) * diff_down)

        self._dip = np.zeros(len(self._close) - (self._window - 1))
        self._dip[0] = pos.dropna()[0: self._window].sum()

        pos = pos.reset_index(drop=True)

        for i in range(1, len(self._dip) - 1):
            self._dip[i] = (
                self._dip[i - 1]
                - (self._dip[i - 1] / float(self._window))
                + pos[self._window + i]
            )

        self._din = np.zeros(len(self._close) - (self._window - 1))
        self._din[0] = neg.dropna()[0: self._window].sum()

        neg = neg.reset_index(drop=True)

        for i in range(1, len(self._din) - 1):
            self._din[i] = (
                self._din[i - 1]
                - (self._din[i - 1] / float(self._window))
                + neg[self._window + i]
            )

    def adx(self) -> pd.Series:
        """Average Directional Index (ADX)

        Returns:
            pandas.Series: New feature generated.tr
        """
        dip = np.zeros(len(self._trs))

        for idx, value in enumerate(self._trs):
            dip[idx] = 100 * (self._dip[idx] / value)

        din = np.zeros(len(self._trs))

        for idx, value in enumerate(self._trs):
            din[idx] = 100 * (self._din[idx] / value)

        directional_index = 100 * np.abs((dip - din) / (dip + din))

        adx_series = np.zeros(len(self._trs))
        adx_series[self._window] = directional_index[0: self._window].mean()

        for i in range(self._window + 1, len(adx_series)):
            adx_series[i] = (
                (adx_series[i - 1] * (self._window - 1)) +
                directional_index[i - 1]
            ) / float(self._window)

        adx_series = np.concatenate((self._trs_initial, adx_series), axis=0)
        adx_series = pd.Series(data=adx_series, index=self._close.index)

        adx_series = self._check_fillna(adx_series, value=20)
        return pd.Series(adx_series, name="adx")

    def adx_pos(self) -> pd.Series:
        """Plus Directional Indicator (+DI)

        Returns:
            pandas.Series: New feature generated.
        """
        dip = np.zeros(len(self._close))
        for i in range(1, len(self._trs) - 1):
            dip[i + self._window] = 100 * (self._dip[i] / self._trs[i])

        adx_pos_series = self._check_fillna(
            pd.Series(dip, index=self._close.index), value=20
        )
        return pd.Series(adx_pos_series, name="adx_pos")

    def adx_neg(self) -> pd.Series:
        """Minus Directional Indicator (-DI)

        Returns:
            pandas.Series: New feature generated.
        """
        din = np.zeros(len(self._close))
        for i in range(1, len(self._trs) - 1):
            din[i + self._window] = 100 * (self._din[i] / self._trs[i])

        adx_neg_series = self._check_fillna(
            pd.Series(din, index=self._close.index), value=20
        )
        return pd.Series(adx_neg_series, name="adx_neg")

def pine_atr(high, low, close, length):

    true_range = pd.Series(np.where(high.shift(1).isna(), high - low,
                                    np.maximum(np.maximum(high - low, (high - close.shift(1)).abs()), (low - close.shift(1)).abs())))

    wilders_moving_avg = true_range.rolling(window=length).sum() / length

    wilders_moving_avg[length:] = wilders_moving_avg[length:].apply(
        lambda x: (wilders_moving_avg[length - 1] * (length - 1) + x) / length)
    return wilders_moving_avg


def supertrendtr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])
    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)
    return tr


def supertrendatr(data, period):
    data['tr'] = supertrendtr(data)
    atr = data['tr'].rolling(period).mean()
    return atr


def Mysupertrend(df, period=10, atr_multiplier=3.0):

    hl2 = (df['high'] + df['low']) / 2
    atrvalue = atr_multiplier * supertrendatr(df, period)
    upperband = hl2 + atrvalue
    lowerband = hl2 - atrvalue
    close = df['close']

    m = len(df.index)
    dir_, trend = [1] * m, [0] * m
    long, short = [npNaN] * m, [npNaN] * m

    for i in range(1, m):
        previous = i - 1
        if close[i] > upperband[previous]:
            dir_[i] = 1
        elif close[i] < lowerband[previous]:
            dir_[i] = 0
        else:
            dir_[i] = dir_[previous]
            if dir_[i] > 0 and lowerband[i] < lowerband[previous]:
                lowerband[i] = lowerband[previous]
            if dir_[i] < 0 and upperband[i] > upperband[previous]:
                upperband[i] = upperband[previous]

        if dir_[i] > 0:
            trend[i] = long[i] = lowerband[i]
        else:
            trend[i] = short[i] = upperband[i]

    df = DataFrame({
            "SUPERT": trend,
            "SUPERTd": dir_,
            "SUPERTl": long,
            "SUPERTs": short,
        }, index=close.index)

    return df


def pandamodifsupertrend(df, period=7, atr_multiplier=3):


    hl2 = (df['high'] + df['low']) / 2
    atrvalue = atr_multiplier * pine_atr(df['high'], df['low'], df['close'], period)
    upperband = hl2 + atrvalue
    lowerband = hl2 - atrvalue
    close = df['close']

    m = len(df.index)
    dir_, trend = [1] * m, [0] * m
    long, short = [npNaN] * m, [npNaN] * m

    for i in range(1, m):
        previous = i - 1
        if close[i] > upperband[previous]:
            dir_[i] = 1
        elif close[i] < lowerband[previous]:
            dir_[i] = 0
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lowerband[i] < lowerband[previous]:
                lowerband[i] = lowerband[previous]
            if dir_[i] < 0 and upperband[i] > upperband[previous]:
                upperband[i] = upperband[previous]

        if dir_[i] > 0:
            trend[i] = long[i] = lowerband[i]
        else:
            trend[i] = short[i] = upperband[i]

    df = DataFrame({
            "SUPERT": trend,
            "SUPERTd": dir_,
            "SUPERTl": long,
            "SUPERTs": short,
        }, index=close.index)

    return df


class HLMSuperSmoth7(IStrategy):
    def version(self) -> str:
        return "v7.0.0"

    class HyperOpt:
        def stoploss_space():

            return [SKDecimal(-0.2, -0.05, decimals=3, name='stoploss')]















    buy_supertrend_period = IntParameter(7, 24, default=10, space='buy', optimize=False, load=False)






















    minimal_roi = {
        "0": 0.132,
        "25": 0.106,
        "70": 0.038,
        "3800": 0

    }

    stoploss = -0.3

    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.094
    trailing_only_offset_is_reached = True

    waitseconds = 60 * 180






    timeframe = '5m'  # 5 dakikalık zaman dilimi
    startup_candle_count: int = 200  # Gereken başlangıç mum sayısı

    informative_timeframe = '5m'







    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "custom_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_price_type": "last",
        "stoploss_on_exchange_interval": 60
    }

    plot_config = {


         'main_plot': {
             "LRC50": {'color': 'green'},
             "EMA13": {'color': 'red'},
             "EMA22": {'color': 'white'},

             },

        'subplots': {

            'indy': {
                    "rsi": {},
                    "adx": {},
                    "adx_pos": {},
                    "fastd": {},
                    "fastk": {},
                    "cti": {},
                    "cmo": {},


                     },

            'UP': {"supertrend_direction": {'color': 'green'}},
            'UPkon': {"sarup": {},

                      "TRTrailingUp": {},
                      "momentdown": {},

                      "LRC50Percent": {},
                      },




             }
        }

    cooldown_lookback = IntParameter(2, 48, default=2, space="protection", optimize=False)
    low_profit_optimize = False
    low_profit_lookback = IntParameter(
        2, 60, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_stop_duration = IntParameter(
        12, 200, default=20, space="protection", optimize=low_profit_optimize)
    low_profit_min_req = DecimalParameter(-0.05, 0.05, default=-0.05,
                                          space="protection", decimals=2, optimize=low_profit_optimize)

    use_custom_stoploss = False

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        prot.append({
            "method": "LowProfitPairs",
            "lookback_period_candles": self.low_profit_lookback.value,
            "trade_limit": 1,
            "stop_duration": int(self.low_profit_stop_duration.value),
            "required_profit": self.low_profit_min_req.value
        })

        return prot

    def super_smooth(self, p, length):
        f = (1.414 * 3.14159) / length
        a = np.exp(-f)
        c2 = 2 * a * np.cos(f)
        c3 = -a * a
        c1 = 1 - c2 - c3
        ssmooth = c1 * (p + np.roll(p, 1)) * 0.5
        ssmooth = ssmooth + c2 * np.roll(ssmooth, 1) + c3 * np.roll(ssmooth, 2)
        return ssmooth

    def super_smooth_macd(self, p, length1, length2, length3):
        ssmooth1 = self.super_smooth(p, length1)
        ssmooth2 = self.super_smooth(p, length2)
        macd = (ssmooth1 - ssmooth2) * 10000000
        ssmooth3 = self.super_smooth(macd, length3)
        return macd, ssmooth3

    def calculate_ATR_trailing_stop(self, df: DataFrame, period: int = 200, keyvalue: float = 3.0) -> DataFrame:

        atr = pine_atr(df['high'], df['low'], df['close'], period)
        nLoss = keyvalue * atr


        m = len(df.index)
        xATRTrailingStop = [0] * m
        for i in range(1, len(df)):
            previous = i - 1
            if df['close'][i] > xATRTrailingStop[previous] and df['close'][previous] > xATRTrailingStop[previous]:
                xATRTrailingStop[i] = max(xATRTrailingStop[previous], df['close'][i] - nLoss[i])
            elif df['close'][i] < xATRTrailingStop[previous] and df['close'][previous] < xATRTrailingStop[previous]:
                xATRTrailingStop[i] = min(xATRTrailingStop[previous], df['close'][i] + nLoss[i])
            elif df['close'][i] > xATRTrailingStop[previous]:
                xATRTrailingStop[i] = df['close'][i] - nLoss[i]
            else:
                xATRTrailingStop[i] = df['close'][i] + nLoss[i]

        return xATRTrailingStop

    def super_smooth(self, p, length):
        f = (1.414 * 3.14159) / length
        a = np.exp(-f)
        c2 = 2 * a * np.cos(f)
        c3 = -a * a
        c1 = 1 - c2 - c3
        ssmooth = c1 * (p + np.roll(p, 1)) * 0.5
        ssmooth = ssmooth + c2 * np.roll(ssmooth, 1) + c3 * np.roll(ssmooth, 2)
        return ssmooth

    def super_smooth_macd(self, p, length1, length2, length3):
        ssmooth1 = self.super_smooth(p, length1)
        ssmooth2 = self.super_smooth(p, length2)
        macd = (ssmooth1 - ssmooth2) * 10000000
        ssmooth3 = self.super_smooth(macd, length3)
        return macd, ssmooth3

    def EWO(self, dataframe, ema_length=5, ema2_length=3):
        df = dataframe.copy()
        ema1 = ta.EMA(df, timeperiod=ema_length)
        ema2 = ta.EMA(df, timeperiod=ema2_length)
        emadif = (ema1 - ema2) / df['close'] * 100
        return emadif

    def PINEHULL(self, dataframe: DataFrame, period: int):

        n = period
        n2ma = 2 * ta.WMA(dataframe['close'], round(n / 2))
        nma = ta.WMA(dataframe['close'], n)
        diff = n2ma - nma
        sqn = round(math.sqrt(n))

        n2ma1 = 2 * ta.WMA(dataframe['close'].shift(), round(n / 2))
        nma1 = ta.WMA(dataframe['close'].shift(), n)
        diff1 = n2ma1 - nma1


        hull1 = ta.WMA(diff, sqn)
        hull2 = ta.WMA(diff1, sqn)

        return hull1, hull2

    def informative_pairs(self):
        if self.timeframe == "1m":
            pairs = self.dp.current_whitelist()
            informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
            return informative_pairs

        return []


    def do_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:




        supertrend_df = pandamodifsupertrend(dataframe,
                                             period=self.buy_supertrend_period.value)


        dataframe['supertrend_direction'] = supertrend_df["SUPERTd"]

        dataframe['xATRTrailingStop'] = self.calculate_ATR_trailing_stop(
            dataframe, period=200, keyvalue=2)

        dataframe['supermacd'],   dataframe['supersmoth'] = self.super_smooth_macd(
            dataframe['close'].values, 8, 13, 3)

        dataframe['LRC50'] = ta.LINEARREG(dataframe['close'], 50, 0)
        dataframe["EMA13"] = ta.EMA(dataframe, timeperiod=13)

        dataframe["EMA50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["EMA200"] = ta.EMA(dataframe, timeperiod=200)

        dataframe["EMA96"] = ta.EMA(dataframe, timeperiod=96)

        dataframe["EMA22"] = ta.EMA(dataframe, timeperiod=22)

        dataframe["EMA22SELL"] = np.where(
         (dataframe['close'] > (dataframe["EMA22"] * 1.01)), 1, 0)

        dataframe['LRC50Percent'] = (dataframe['EMA13'] - dataframe['LRC50']) / dataframe['LRC50']

        dataframe['EMA200Percent'] = (
            dataframe['close'] - dataframe['EMA200']) / dataframe['EMA200']



        dataframe['LRC50Percent'] = dataframe['LRC50Percent'].astype(float)

        dataframe["LRC50DISSUC"] = np.where(dataframe['LRC50Percent'] >= 0.01, 1, 0)

        dataframe["EMASUCCES"] = np.where((dataframe['close'] > dataframe['EMA13'])  # ((dataframe['EMA13'] > dataframe['LRC50']) |


                                          & (dataframe['EMA13'] > dataframe['LRC50']), 1, 0)





        dataframe['EWO'] = self.EWO(dataframe, 50, 200)

        dataframe["EWOSUCCES"] = np.where(
         (dataframe['EWO'] >= 0.3) &

            (dataframe['EWO'] < 3.0), 1, 0)

        dataframe["MOMENTUMTREND"] = ta.LINEARREG(
            dataframe['close'] - ta.SMA((dataframe['high'] + dataframe['low']) / 2, 20), 20) * 1000

        dataframe['rsi'] = ta.RSI(dataframe, 14)
        dataframe['rsi'] = dataframe['rsi'].fillna(0).astype('int')
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe["macd_sell"] = np.where(
             (dataframe['macd'] < dataframe['macdsignal']), 1, 0)




        stoch_fast = qtpylib.stoch(dataframe, window=14, fast=True)
        dataframe['fastd'] = stoch_fast['fast_d']
        dataframe['fastk'] = stoch_fast['fast_k']

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        dataframe['cti_mean'] = dataframe['cti'].rolling(20).mean()

        dataframe['cmo'] = ta.CMO(dataframe, timeperiod=24)

        dataframe['cmopos'] = dataframe['cmo'] - dataframe['cmo'].shift()

        indicator_adx = TAADXIndicator(
            high=dataframe["high"], low=dataframe["low"], close=dataframe["close"]
        )
        dataframe["adx"] = indicator_adx.adx().fillna(0).astype('int')
        dataframe["adx_pos"] = indicator_adx.adx_pos().fillna(0).astype('int')
        dataframe["trend_adx_neg"] = indicator_adx.adx_neg()

        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['TEMA'] = ta.TEMA(dataframe, timeperiod=12)

        dataframe["hlc3"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3

        esa = ta.EMA(dataframe["hlc3"], 10)

        d = ta.EMA((dataframe["hlc3"] - esa).abs(), 10)
        ci = (dataframe["hlc3"] - esa).div(0.0015 * d)
        tci = ta.EMA(ci, 21)

        wt1 = tci
        wt2 = ta.SMA(np.nan_to_num(wt1), 4)

        dataframe['wt1'], dataframe['wt2'] = wt1, wt2

        dataframe['volume_mean_22'] = dataframe['volume'].rolling(22).mean().shift(1)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=60, stds=1.4)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['sarup'] = np.where(

            (dataframe['sar'] < dataframe['close']), 1, 0)

        dataframe['wavetrend'] = np.where(

            (dataframe['wt1'] > dataframe['wt2']), 1, 0)

        dataframe['TRTrailingUp'] = np.where(

            (dataframe['close'] > dataframe['xATRTrailingStop']), 1, 0)

        dataframe['momentumup'] = np.where(

          (dataframe["MOMENTUMTREND"] >= 0.01), 1, 0)

        dataframe['momentdown'] = np.where(

          (dataframe["MOMENTUMTREND"] > 10.0) &
          (dataframe["adx_pos"] < 30), 1, 0)

        dataframe['ctihigh'] = np.where(

          (dataframe['cti'] >= 0.7)

          | ((dataframe['cti'].shift() < 0.0) & (dataframe['cti'] > 3.0)

             ), 1, 0)

        dataframe['ctiadx'] = np.where(

          (dataframe['cti'] >= 0.6) &

          (dataframe["adx"] < 30) &
          (dataframe["adx_pos"] < 30), 1, 0)

        dataframe['fastdadx'] = np.where(

         ((dataframe['fastd'] >= 75) &

          (dataframe["adx"] < 20)) |

            ((dataframe['adx_pos'] >= 50) &

             (dataframe["adx"] < 20)) |

            ((dataframe['cmo'].shift() < 0.0) &

             (dataframe['cmo'] >= 30)

             ), 1, 0)

        dataframe['RSISUCCES'] = np.where(


          (dataframe['rsi'] > dataframe['rsi'].shift()), 1, 0)

        dataframe["aroon_up"], dataframe["aroon_down"] = ta.AROON(
            dataframe["high"], dataframe["low"], timeperiod=13)

        dataframe["aroon_down"] = dataframe["aroon_down"].fillna(0).astype('int')

        indicator_aroon = TA_AroonIndicator(close=dataframe["close"], window=25)
        dataframe["ta_aroon_up"] = indicator_aroon.aroon_up()
        dataframe["ta_aroon_down"] = indicator_aroon.aroon_down()
        dataframe["ta_aroon_ind"] = indicator_aroon.aroon_indicator()

        dataframe['middlebandPercent'] = (
            dataframe['close'] - dataframe['bb_middleband']) / dataframe['bb_middleband']

        dataframe['midlebandcti'] = np.where(

          ((dataframe['cti'] >= 0.6) &

           (dataframe['middlebandPercent'] >= 0.02) &
           (dataframe["aroon_up"] >= 30)) |

            (dataframe["aroon_up"] >= 88), 1, 0)






        dataframe["cri"] = TACumulativeReturnIndicator(
            close=dataframe["close"]
        ).cumulative_return()

        dataframe["wr"] = TAWilliamsRIndicator(
            high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], lbp=14
            ).williams_r()

        dataframe["tavolume_adi"] = TAAccDistIndexIndicator(
            high=dataframe["high"], low=dataframe["low"], close=dataframe["close"], volume=dataframe["volume"]
        ).acc_dist_index()

        dataframe["tavolume_obv"] = TAOnBalanceVolumeIndicator(
            close=dataframe["close"], volume=dataframe["volume"]
        ).on_balance_volume()

        dataframe['wrbad'] = np.where(



           ((dataframe["wr"] < -7.0) &

            (

                (dataframe["adx_pos"] > 60.0)
                | (dataframe["cmo"] > 60.0)


               )

            ) |
            (
            (dataframe['EWO'] < 0.5) &
            ((dataframe["wr"] < -3.0) | (dataframe["ta_aroon_ind"] > 50))


            ), 1, 0)

        dataframe['volumebad'] = np.where(

           ((dataframe["tavolume_obv"] < 0.0) &
            (dataframe["tavolume_adi"] < 0.0)) |

            (
              (dataframe["cti"] > 0.5) &
            (dataframe["tavolume_obv"] < 0.0)


           ), 1, 0)

        dataframe['cribad'] = np.where(

           ((dataframe["cri"] < 0.0) &
            (

            (dataframe["adx"] < 20) | (dataframe["tavolume_obv"] < 0.0)
            | (dataframe["ta_aroon_ind"] > dataframe["ta_aroon_down"])

            )


            ) |

            (dataframe["cri"] > 80.0), 1, 0)

        dataframe['badhigh'] = np.where(

            (dataframe["cti"] > 0.5) &
            (dataframe['fastd'] >= 80) &
            (dataframe['fastk'] >= 90) &
            (dataframe["adx"] < 30) &
            (dataframe["cmo"] > 35) &
            (dataframe["adx_pos"] >= 35), 1, 0)

        dataframe['volumebad1'] = np.where(

            (dataframe["tavolume_adi"] < 0.0) &

            (dataframe["cti"] > 0.5) &
            (dataframe['fastd'] >= 70) &
            (dataframe['fastk'] >= 70) &
            (dataframe["adx_pos"] >= 30) &
            (dataframe["rsi_fast"] >= 75) &

            (dataframe["cmo"] >= 30), 1, 0)

        dataframe['EMA200BAD'] = np.where(
            (dataframe['EMA200Percent'] > 0.03) &
            (dataframe["MOMENTUMTREND"] < 0.5) &
            (dataframe["fastk"] > 90) &
            (dataframe["rsi_fast"] > 85), 1, 0)


        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=14)


        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)



        dataframe['lambo2sell'] = (










            (dataframe['close'] > (dataframe["EMA13"] * 1.01))
            |
            (dataframe['close'] > (dataframe["LRC50"] * 1.01))
            )

        dataframe["lambo2aronbad"] = (

             (dataframe['middlebandPercent'] < 0.0) &
           (((dataframe['aroon_down'] < 90) &
             (dataframe['ta_aroon_up'] < 90))
                 | (
                    (dataframe['adx'] > 30) &
                    (dataframe['fastd'] > 30) &
                    (dataframe['fastk'] > 30)


                  ))


        )




        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.timeframe == "1m":
            informative = self.dp.get_pair_dataframe(
                pair=metadata['pair'], timeframe=self.informative_timeframe)

            informative = self.do_indicators_5m(informative.copy(), metadata)



            dataframe = merge_informative_pair(
                dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)

            skip_columns = [(s + "_" + self.informative_timeframe)
                            for s in ['open', 'high', 'low', 'close', 'volume']]

            dataframe.rename(columns=lambda s: s.replace("_{}".format(
                self.informative_timeframe), "") if (not s in skip_columns) else s, inplace=True)
        else:
            dataframe = self.do_indicators_5m(dataframe, metadata)







        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions_long = []

        dataframe.loc[:, 'enter_tag'] = ''

        buy_1 = (

             (dataframe['rsi'] > dataframe['rsi'].shift()) &

             (dataframe['cribad'] < 1) &
             (dataframe['wrbad'] < 1) &


             (dataframe['volumebad'] < 1) &
             (dataframe['volumebad1'] < 1) &
            (dataframe['fastdadx'] < 1) &
            (dataframe['badhigh'] < 1) &







            (dataframe['momentumup'] > 0) &






            (dataframe["EWOSUCCES"] > 0) &
            (dataframe["EMA22SELL"] > 0) &

            (dataframe["EMASUCCES"] > 0) &

            (dataframe["LRC50DISSUC"] < 1) &




            (dataframe['ctihigh'] < 1) &
            (dataframe['ctiadx'] < 1) &
            (dataframe['aroon_down'] == 100) &
            (dataframe['ta_aroon_up'] == 100) &


            (dataframe["midlebandcti"] < 1) &


            (dataframe['momentdown'] < 1) &




            (dataframe['cmo'] >= 5) &
            (dataframe['cmo'] < 45) &
            (dataframe['cmopos'] < 40) &

            (dataframe['rsi'] < 90) &
            (dataframe['adx'] < 60) &


            (dataframe['fastk'] > dataframe['fastd']) &


            (dataframe['fastd'] < 90) &
            (dataframe['fastd'].shift() < 90) &
            (dataframe['fastk'] < 100) &
            (dataframe['sarup'] > 0) &
            (dataframe['volume'] > dataframe['volume_mean_22']) &
            (dataframe['supertrend_direction'] > 0) &


            (dataframe['wavetrend'] > 0) &
            (dataframe['supermacd'] > dataframe['supersmoth']) &
            (dataframe['TRTrailingUp'] > 0) &




            (dataframe['volume'] > 0)
        )

        lambo2 = (


            (dataframe['lambo2sell'] == False) &
            (dataframe['lambo2aronbad'] == False) &
            (dataframe['TEMA'] > dataframe["hlc3"]) &



            (dataframe['rsi_fast'] < 35) &
            (dataframe['close'] < (dataframe["ema_12"] * 0.987)) &
            (dataframe['EWO'] > 4.179) &
            (dataframe['rsi'] < 39) &


            (dataframe['volume'] > 0)

        )

        buy_6 = (


            (dataframe['TRTrailingUp'] > 0) &
            (dataframe['TRTrailingUp'].shift() < 1) &


            (dataframe['supertrend_direction'] > 0) &
            (dataframe['fastk'].shift() < dataframe['fastd'].shift()) &

            (dataframe['fastk'].shift() < 30) &
            (dataframe['fastd'].shift() < 30) &
            (dataframe['fastd'] < 50) &
            (dataframe['fastk'] > 50) &
            (dataframe['cti'] < 0.0) &
            (dataframe['cmo'] < 10) &
            (dataframe['EWO'] > -3.0) &
            (dataframe['rsi'] < 58) &
            (dataframe['adx'] >= 25) &
            (dataframe['volumebad'] < 1) &


            (dataframe["EMA13"] < dataframe["close"]) &
            (dataframe["LRC50"] < dataframe["close"]) &






            (dataframe['volume'] > 0)

        )

        buy_7 = (


            (
                ((dataframe['TRTrailingUp'] > 0) &
                 (dataframe['TRTrailingUp'].shift() < 1))
                |


                (
                    (dataframe['supertrend_direction'] > 0) &
                    (dataframe['supertrend_direction'].shift() < 1)

                    )) &


            (dataframe['fastk'].shift() < dataframe['fastd'].shift()) &

            (dataframe['fastk'].shift() < 30) &
            (dataframe['fastd'].shift() < 30) &
            (dataframe['fastd'] < 60) &
            (dataframe['fastk'] > 80) &
            (dataframe['cti'] < 0.0) &
            (dataframe['cmo'] < 10) &
            (dataframe['EWO'] > -3.0) &
            (dataframe['rsi'] < 58) &
            (dataframe['adx'] >= 25) &
            (dataframe['volumebad'] < 1) &
            (dataframe["LRC50"] < dataframe["EMA13"]) &
            (dataframe['volume'] > 0)

        )

        buy_3 = (



            ((dataframe['rsi'].shift() < 68) & (dataframe['rsi'] > 80)) &
            ((dataframe['cmo'].shift() < 35) & (dataframe['cmo'] > 50)) &
            (dataframe['cribad'] < 1) &
            (dataframe['wrbad'] < 1) &
            (dataframe['volumebad'] < 1) &
            (dataframe['fastdadx'] < 1) &
            (dataframe['ta_aroon_up'] == 100) &
            (dataframe['aroon_down'] == 100) &
            (dataframe["macd_sell"] < 1) &
            (dataframe['EMA200BAD'] < 1) &


            (dataframe['adx'] < 35) &

            (dataframe['cmopos'] < 50) &
            (dataframe['LRC50Percent'] < 0.01) &

            (dataframe['supermacd'] > dataframe['supersmoth']) &

            (dataframe['volume'] > 0) &
            (dataframe["EMASUCCES"] > 0) &
            (dataframe["EWOSUCCES"] > 0) &
            (dataframe['ctihigh'] < 1) &
            (dataframe['cti_mean'] < 0.6) &
            (dataframe['sarup'] > 0) &
            (dataframe['TRTrailingUp'] > 0)
        )
























        conditions_long.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] = 'buy_1'

        conditions_long.append(buy_3)
        dataframe.loc[buy_3, 'enter_tag'] += 'buy_3'



        conditions_long.append(lambo2)
        dataframe.loc[lambo2, 'enter_tag'] += 'lambo2'
        conditions_long.append(buy_6)
        dataframe.loc[buy_6, 'enter_tag'] += 'buy_6'
        conditions_long.append(buy_7)
        dataframe.loc[buy_7, 'enter_tag'] += 'buy_7'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'enter_long'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:














        return dataframe

    debug = {}

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:





        return True

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        self.debug[pair] = last_candle




        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:





        return True
        profit_or_loss = rate - trade.open_rate

        profit_or_loss_percentage = (profit_or_loss / trade.open_rate) * 100

        last_candle = self.debug[pair]

        lost = True
        kazyaz = True
        kaypermax = 20
        kaypermin = 2


        if (exit_reason == "roi" and profit_or_loss_percentage < 0.5) or (kazyaz and profit_or_loss_percentage > kaypermin and profit_or_loss_percentage < kaypermax) or (lost and profit_or_loss_percentage <= 0.0):
            for column_name, column_value in last_candle.items():
                print(f"{pair} {column_name}: {column_value}")

            print(f"{exit_reason} {pair} %{profit_or_loss_percentage}")
            print("/////////////////////////////////////////////////////////")






        return True

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        profit_or_loss = current_rate - trade.open_rate

        profit_or_loss_percentage = (profit_or_loss / trade.open_rate) * 100

        higbuy = bool(trade.enter_tag == "buy_3" or trade.enter_tag ==
                      "buy_4" or trade.enter_tag == "buy_3buy_4" or trade.enter_tag == "buy_6")

        supersmothstatus = bool(trade.enter_tag == "buy_6" or trade.enter_tag == "buy_7")

        if profit_or_loss_percentage > 0.3:
            if last_candle["TRTrailingUp"] < 1:
                if (current_time - trade.open_date_utc).seconds >= self.waitseconds:
                    if last_candle["supersmoth"] > last_candle["supermacd"]:



                        return 'timesupermacdPercent_' + trade.enter_tag

                    if last_candle['fastk'] < last_candle['fastd']:



                        return 'timefastkcti' + trade.enter_tag

        if profit_or_loss_percentage > 0.5:

            if trade.enter_tag == "lambo2":
                if last_candle['lambo2sell'] == True:
                    return 'lambo2sell' + trade.enter_tag
                return False

            if higbuy == False:




                if last_candle["adx"] >= 65:
                    return 'adxsell_' + trade.enter_tag

            if last_candle['fastk'] < last_candle['fastd'] and last_candle["cti"] >= 0.7 and last_candle['RSISUCCES'] < 1:
                if last_candle['supertrend_direction'] < 1 and last_candle["rsi"] > 58 and last_candle["LRC50"] > last_candle["EMA13"]:
                    return 'fastkcti' + trade.enter_tag

            if supersmothstatus == False and last_candle["supersmoth"] > last_candle["supermacd"] and last_candle['RSISUCCES'] < 1 and last_candle["LRC50"] > last_candle["EMA13"]:

                return 'supermacdPercent_' + trade.enter_tag

            if last_candle["cti"] >= 0.9:
                return 'ctisell_' + trade.enter_tag

            if last_candle["TRTrailingUp"] < 1:



                if last_candle["LRC50"] * 1.01 > last_candle["close"]:
                    return 'LRC50sell_' + trade.enter_tag

                if last_candle["EMA13"] * 1.01 > last_candle["close"]:
                    return 'EMA13SELL_' + trade.enter_tag

                if last_candle["EMA22SELL"] < 1:
                    return 'EMA22SELL_' + trade.enter_tag

                if last_candle['close'] > last_candle['bb_middleband'] * 1.01:
                    return 'bb_middlebandsell_' + trade.enter_tag
        else:

            if profit_or_loss_percentage < -6:
                if (current_time - trade.open_date_utc).days >= 4:
                    return 'timesell' + trade.enter_tag

                if last_candle["supersmoth"] > last_candle["supermacd"] and last_candle['supertrend_direction'] < 1 and last_candle['sarup'] < 1:
                    return 'supersmoth' + trade.enter_tag

        return False
