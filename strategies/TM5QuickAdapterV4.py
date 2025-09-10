import logging
from functools import reduce
import datetime
from datetime import timedelta
import talib.abstract as ta
from pandas import DataFrame, Series
from technical import qtpylib
from typing import Optional
from freqtrade import data
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from scipy.signal import argrelextrema
import numpy as np
import pandas_ta as pta
import math

from lib.tm5_prediction_storage import TM5PredictionStorage

logger = logging.getLogger(__name__)

"""
The following strategy is released to sponsors of the non-profit FreqAI open-source project.
If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
We use sponsor money to help stimulate new features and to pay for running these public
experiments, with a an objective of helping the community make smarter choices in their
ML journey.

This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
returns. The goal is to demonstrate gratitude to people who support the project and to
help them find a good starting point for their own creativity.

If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

https://github.com/sponsors/robcaulk
"""


class TM5QuickAdapterV4(IStrategy):

    position_adjustment_enable = False

    stoploss = -0.02

    accuracy_scores = DataFrame()

    order_types = {
        "entry": "limit",
        "exit": "market",
        "emergency_exit": "market",
        "force_exit": "market",
        "force_entry": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 120,
    }

    max_entry_position_adjustment = 1

    max_dca_multiplier = 2

    minimal_roi = {"0": 0.04, "5000": -1}

    process_only_new_candles = True

    can_short = True

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "&s-extrema": {
                    "color": "#f53580",
                    "type": "line"
                },
                "&s-minima_sort_threshold": {
                    "color": "#f66151",
                    "type": "line"
                },
                "&s-maxima_sort_threshold": {
                    "color": "#8ff0a4",
                    "type": "line"
                }
            },
            "range_est": {
                "&-s_max": {
                    "color": "#a29db9",
                    "type": "line"
                },
                "&-s_min": {
                    "color": "#ac7fc",
                    "type": "line"
                }
            },
            "truth": {
                "maxima-exit": {
                    "color": "#8ff0a4",
                    "type": "bar"
                },
                "minima-exit": {
                    "color": "#f66151",
                    "type": "bar"
                }
            },
            "reddit": {
                "%%-social_volume_reddit/bitcoin": {
                    "color": "#75e918"
                }
            },
            "nvt": {
                "%%-nvt_5min/bitcoin": {
                    "color": "#be2306"
                }
            }
        }
    }

    @property
    def PREDICT_STORAGE_ENABLED(self):
        return self.config["sagemaster"].get("PREDICT_STORAGE_ENABLED", False)

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2,
            },

        ]

    use_exit_signal = True
    startup_candle_count: int = 80

    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    def bot_start(self, **kwargs) -> None:
        print("bot_start")

        if (self.PREDICT_STORAGE_ENABLED):
            self.ps = TM5PredictionStorage(connection_string=self.config["sagemaster"].get("PREDICT_STORAGE_CONN_STRING"))


    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(dataframe['close'], length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
        dataframe["%-cti-period"] = pta.cti(dataframe['close'], length=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(
            dataframe['close'], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe[f"%-atr-period"] / \
            dataframe['close'] * 1000
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, **kwargs):
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-obv"] = ta.OBV(dataframe)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=14, stds=2.2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["%-bb_width"] = (dataframe["bb_upperband"] -
                                   dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['%-distema50'] = get_distance(
            dataframe['close'], dataframe['ema_50'])
        dataframe['%-distema12'] = get_distance(
            dataframe['close'], dataframe['ema_12'])
        dataframe['%-distema26'] = get_distance(
            dataframe['close'], dataframe['ema_26'])
        macd = ta.MACD(dataframe)
        dataframe['%-macd'] = macd['macd']
        dataframe['%-macdsignal'] = macd['macdsignal']
        dataframe['%-macdhist'] = macd['macdhist']
        dataframe['%-dist_to_macdsignal'] = get_distance(
            dataframe['%-macd'], dataframe['%-macdsignal'])
        dataframe['%-dist_to_zerohist'] = get_distance(
            0, dataframe['%-macdhist'])

        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['%-vwap_width'] = ((dataframe['vwap_upperband'] -
                                     dataframe['vwap_lowerband']) / dataframe['vwap_middleband']) * 100
        dataframe = dataframe.copy()
        dataframe['%-dist_to_vwap_upperband'] = get_distance(
            dataframe['close'], dataframe['vwap_upperband'])
        dataframe['%-dist_to_vwap_middleband'] = get_distance(
            dataframe['close'], dataframe['vwap_middleband'])
        dataframe['%-dist_to_vwap_lowerband'] = get_distance(
            dataframe['close'], dataframe['vwap_lowerband'])
        dataframe['%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%-wick'] = (dataframe['high'] - dataframe['close']).abs()
        pp = pivots_points(dataframe)
        dataframe['pivot'] = pp['pivot']
        dataframe['r1'] = pp['r1']
        dataframe['s1'] = pp['s1']
        dataframe['r2'] = pp['r2']
        dataframe['s2'] = pp['s2']
        dataframe['r3'] = pp['r3']
        dataframe['s3'] = pp['s3']
        dataframe['rawclose'] = dataframe['close']
        dataframe['%-dist_to_r1'] = get_distance(
            dataframe['close'], dataframe['r1'])
        dataframe['%-dist_to_r2'] = get_distance(
            dataframe['close'], dataframe['r2'])
        dataframe['%-dist_to_r3'] = get_distance(
            dataframe['close'], dataframe['r3'])
        dataframe['%-dist_to_s1'] = get_distance(
            dataframe['close'], dataframe['s1'])
        dataframe['%-dist_to_s2'] = get_distance(
            dataframe['close'], dataframe['s2'])
        dataframe['%-dist_to_s3'] = get_distance(
            dataframe['close'], dataframe['s3'])
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]
        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["day_of_week"] = (dataframe["date"].dt.dayofweek)
        dataframe["hour_of_day"] = (dataframe["date"].dt.hour)
        dataframe['day_of_week_norm'] = 2 * math.pi * \
            dataframe['day_of_week'] / dataframe['day_of_week'].max()
        dataframe['hour_of_day_norm'] = 2 * math.pi * \
            dataframe['hour_of_day'] / dataframe['hour_of_day'].max()

        dataframe['%%-day_of_week_cos'] = np.cos(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_cos'] = np.cos(dataframe['hour_of_day_norm'])
        dataframe['%%-day_of_week_sin'] = np.sin(dataframe['day_of_week_norm'])
        dataframe['%%-hour_of_day_sin'] = np.sin(dataframe['hour_of_day_norm'])
        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&s-extrema"] = 0
        kernel = self.freqai_info["feature_parameters"]["label_period_candles"]
        min_peaks = argrelextrema(
            dataframe["low"].values, np.less,
            order=kernel
        )
        max_peaks = argrelextrema(
            dataframe["high"].values, np.greater,
            order=kernel
        )
        for mp in min_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = -1
        for mp in max_peaks[0]:
            dataframe.at[mp, "&s-extrema"] = 1
        dataframe["minima-exit"] = np.where(
            dataframe["&s-extrema"] == -1, 1, 0)
        dataframe["maxima-exit"] = np.where(dataframe["&s-extrema"] == 1, 1, 0)
        dataframe['&s-extrema'] = dataframe['&s-extrema'].rolling(
            window=5, win_type='gaussian', center=True).mean(std=0.5)

        dataframe['&-s_max'] = dataframe["close"].shift(-kernel).rolling(
            kernel).max()/dataframe["close"] - 1
        dataframe['&-s_min'] = dataframe["close"].shift(-kernel).rolling(
            kernel).min()/dataframe["close"] - 1

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1,
        )

        dataframe["minima_sort_threshold"] = dataframe["&s-minima_sort_threshold"]
        dataframe["maxima_sort_threshold"] = dataframe["&s-maxima_sort_threshold"]

        last_candle = dataframe.iloc[-1].squeeze()
        if (self.PREDICT_STORAGE_ENABLED and self.ps and (last_candle['do_predict'] == 1)):
            self.ps.add_prediction(
                model_version=self.freqai.identifier,
                pair=metadata['pair'],
                timeframe=self.timeframe,
                extrema=last_candle['&s-extrema'],
                minima_sort_threshold=last_candle['minima_sort_threshold'],
                maxima_sort_threshold=last_candle['maxima_sort_threshold'],
                range_max=last_candle['&-s_max'],
                range_min=last_candle['&-s_min'],
                di_values=last_candle['DI_values'],
                di_cutoff=last_candle['DI_cutoff'],
                candle_time=last_candle['date']
            )

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_tag'] = ''
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["DI_catch"] == 1,
            df["&s-extrema"] < df["minima_sort_threshold"],

        ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), [
                    "enter_long", "enter_tag"]
            ] = (1, "long")

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["DI_catch"] == 1,
            df["&s-extrema"] > df["maxima_sort_threshold"],

        ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), [
                    "enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        return df

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ):

        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        trade_date = timeframe_to_prev_date(
            self.timeframe, (trade.open_date_utc -
                             timedelta(minutes=int(self.timeframe[:-1])))
        )
        trade_candle = dataframe.loc[(dataframe["date"] == trade_date)]

        if trade_candle.empty:
            return None
        trade_candle = trade_candle.squeeze()

        entry_tag = trade.enter_tag

        trade_duration = (current_time - trade.open_date_utc).seconds / 60

        trade_max = trade_candle["&-s_max"]
        trade_min = trade_candle["&-s_min"]

        if trade_duration > 1000:
            return "trade expired"

        if entry_tag == "long" and current_profit > trade_max:
            return "Hit long target"

        if entry_tag == "long" and current_profit < -trade_max:
            return "Missed long target"

        if entry_tag == "short" and current_profit > abs(trade_min):
            return "Hit short target"

        if entry_tag == "short" and current_profit < trade_min:
            return "Missed short target"

        if last_candle["DI_catch"] == 0  and current_profit < 0:
            return "Outlier detected"

        if (
            last_candle["&s-extrema"] < last_candle["minima_sort_threshold"]
            and entry_tag == "short"
        ):
            return "minimia_detected_short"

        if (
            last_candle["&s-extrema"] > last_candle["maxima_sort_threshold"]
            and entry_tag == "long"
        ):
            return "maxima_detected_long"

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:

        open_trades = Trade.get_trades(trade_filter=Trade.is_open.is_(True))

        num_shorts, num_longs = 0, 0
        for trade in open_trades:
            if "short" in trade.enter_tag:
                num_shorts += 1
            elif "long" in trade.enter_tag:
                num_longs += 1

        if side == "long" and num_longs >= 5:
            return False

        if side == "short" and num_shorts >= 5:
            return False

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True


def top_percent_change(dataframe: DataFrame, length: int) -> float:
    """
    Percentage change of the current close from the range maximum Open price
    :param dataframe: DataFrame The original OHLC dataframe
    :param length: int The length to look back
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')



def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def EWO(dataframe, sma_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def get_distance(p1, p2):
    return abs((p1) - (p2))
