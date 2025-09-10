import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import (
    merge_informative_pair,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    stoploss_from_open,
)
from functools import reduce
import logging

logger = logging.getLogger(__name__)

class DiscoveredAlphaV2(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.151,
        "27": 0.036,
        "73": 0.011,
        "176": 0
    }
    stoploss = -0.167

    timeframe = "5m"
    informative_timeframe = "1h"
    informative_daily = "1d"

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.001
    ignore_roi_if_buy_signal = True

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.2
    trailing_stop_positive_offset = 0.234

    use_custom_stoploss = True
    process_only_new_candles = True

    startup_candle_count = 200

    buy_params = {
        "base_nb_candles_buy": 15,
        "ewo_high": 2.556,
        "ewo_low": -11.988,
        "fast_ewo": 27,
        "low_offset": 0.95,
        "rsi_buy": 37,
        "slow_ewo": 141,
        "smaoffset_buy_condition_0_enable": True
    }

    sell_params = {
        "base_nb_candles_sell": 79,
        "high_offset": 1.056,
        "smaoffset_sell_condition_0_enable": False
    }

    smaoffset_buy_condition_0_enable = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    smaoffset_sell_condition_0_enable = CategoricalParameter([True, False], default=False, space="sell", optimize=True)
    base_nb_candles_buy = IntParameter(5, 80, default=15, space="buy", optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=79, space="sell", optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=0.95, space="buy", optimize=True)
    high_offset = DecimalParameter(0.99, 1.1, default=1.056, space="sell", optimize=True)
    fast_ewo = IntParameter(10, 50, default=27, space="buy", optimize=True)
    slow_ewo = IntParameter(100, 200, default=141, space="buy", optimize=True)
    ewo_low = DecimalParameter(-20.0, -8.0, default=-11.988, space="buy", optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=2.556, space="buy", optimize=True)
    rsi_buy = IntParameter(30, 70, default=37, space="buy", optimize=True)

    def custom_stoploss(
        self, pair: str, trade: "Trade", current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:


        if current_profit > 0:
            return 0.99
        else:
            trade_time_50 = trade.open_date_utc + timedelta(minutes=240)

            if current_time > trade_time_50:
                try:
                    number_of_candle_shift = int((current_time - trade_time_50).total_seconds() / 300)
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    candle = dataframe.iloc[-number_of_candle_shift].squeeze()

                    sma_200_dec = bool(candle["sma_200_dec"])
                    sma_200_dec_1h = bool(candle["sma_200_dec_1h"])

                    if sma_200_dec and sma_200_dec_1h:
                        return 0.01
                    if candle["rsi_1h"] < 30:
                        return 0.99

                    if candle["close"] > candle["ema_200"]:
                        if current_rate * 1.025 < candle["open"]:
                            return 0.01

                    if current_rate * 1.015 < candle["open"]:
                        return 0.01

                except IndexError as error:
                    return 0.1

        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        informative_pairs += [(pair, self.informative_daily) for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)

        informative_1h["ema_50"] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h["ema_100"] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h["ema_200"] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h["sma_200"] = ta.SMA(informative_1h, timeperiod=200)
        informative_1h["sma_200_dec"] = informative_1h["sma_200"] < informative_1h["sma_200"].shift(20)
        informative_1h["rsi"] = ta.RSI(informative_1h, timeperiod=14)

        ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        informative_1h["ssl_down"] = ssl_down_1h
        informative_1h["ssl_up"] = ssl_up_1h
        informative_1h["ssl-dir"] = np.where(ssl_up_1h > ssl_down_1h, "up", "down")

        return informative_1h

    def informative_daily_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        informative_daily = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_daily)

        informative_daily["ema_50"] = ta.EMA(informative_daily, timeperiod=50)
        informative_daily["ema_100"] = ta.EMA(informative_daily, timeperiod=100)
        informative_daily["ema_200"] = ta.EMA(informative_daily, timeperiod=200)
        informative_daily["rsi"] = ta.RSI(informative_daily, timeperiod=14)

        ssl_down_daily, ssl_up_daily = SSLChannels(informative_daily, 20)
        informative_daily["ssl_down"] = ssl_down_daily
        informative_daily["ssl_up"] = ssl_up_daily

        return informative_daily

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb_40 = qtpylib.bollinger_bands(dataframe["close"], window=40, stds=2)
        dataframe["lower"] = bb_40["lower"]
        dataframe["mid"] = bb_40["mid"]
        dataframe["bbdelta"] = (bb_40["mid"] - dataframe["lower"]).abs()
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift(1)).abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        dataframe["volume_mean_slow"] = dataframe["volume"].rolling(window=30, min_periods=1).mean()

        dataframe["ema_12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_26"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        dataframe["sma_5"] = ta.EMA(dataframe, timeperiod=5)
        dataframe["sma_200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["sma_200_dec"] = dataframe["sma_200"] < dataframe["sma_200"].shift(20)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        for val in self.base_nb_candles_buy.range:
            if val > 1:
                dataframe[f"ma_buy_{val}"] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            if val > 1:
                dataframe[f"ma_sell_{val}"] = ta.EMA(dataframe, timeperiod=val)

        dataframe["EWO"] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True
        )

        informative_daily = self.informative_daily_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_daily, self.timeframe, self.informative_daily, ffill=True
        )

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        dataframe.loc[:, "smaoffset_buy_condition_0_enable"] = False
        dataframe.loc[:, "buy_tag"] = ''
        dataframe.loc[:, "conditions_count"] = 0

        dataframe["ma_buy"] = dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"] * self.low_offset.value

        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ma_buy"])
                & (dataframe["EWO"] > self.ewo_high.value)
                & (dataframe["rsi"] < self.rsi_buy.value + 10)  # Less restrictive RSI
                & (self.smaoffset_buy_condition_0_enable.value == True)
            ),
            ['smaoffset_buy_condition_0_enable', 'buy_tag']] = (True, 'buy_signal_smaoffset_0')

        dataframe.loc[:, "conditions_count"] = dataframe["smaoffset_buy_condition_0_enable"].astype(int)

        conditions.append(dataframe["conditions_count"] >= 1)
        conditions.append(dataframe["volume"].gt(0))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "buy"] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe["ma_sell"] = dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"] * self.high_offset.value

        if self.smaoffset_sell_condition_0_enable.value:
            conditions.append(
                (
                    (qtpylib.crossed_below(dataframe["close"], dataframe["ma_sell"]))
                    & (dataframe["volume"] > 0)
                )
            )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe

def SSLChannels(dataframe: DataFrame, length: int = 7) -> (np.ndarray, np.ndarray):
    """
    SSL Channels: Calculate the SSL Channels indicator.

    Parameters:
    dataframe: DataFrame - The OHLCV data.
    length: int - Rolling window length for SMA calculation.

    Returns:
    Tuple: Arrays containing the sslDown and sslUp channels.
    """
    df = dataframe.copy()

    df["ATR"] = ta.ATR(df, timeperiod=14)

    df["smaHigh"] = df["high"].rolling(window=length, min_periods=length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(window=length, min_periods=length).mean() - df["ATR"]

    df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.nan))

    df["hlv"] = df["hlv"].ffill()

    df["sslDown"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["sslUp"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])

    return df["sslDown"].values, df["sslUp"].values

def EWO(dataframe: DataFrame, ema_length: int = 5, ema2_length: int = 35) -> np.ndarray:
    if ema_length <= 1 or ema2_length <= 1:
        raise ValueError("Les périodes EMA doivent être supérieures à 1.")
    
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / dataframe["close"] * 100
    return emadif
