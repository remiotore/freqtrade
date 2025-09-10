
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
import talib.abstract as ta
from datetime import datetime, timedelta

import numpy as np

from freqtrade.rpc import RPCMessageType

import freqtrade.vendor.qtpylib.indicators as qtpylib
import os



class SpringAlarm(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '1h'

    alarm_emitted = dict()

    start_alarm_check_minute = 50

    volume_condition_off_range_top_limit_multiplier = 1.05
    volume_condition_off_range_bottom_limit_multiplier = 0.85

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if pair not in self.alarm_emitted:
            self.alarm_emitted[pair] = False

        if datetime.now().minute < self.start_alarm_check_minute:
            self.alarm_emitted[pair] = False

        self.add_volume_condition(dataframe)
        self.add_volume_condition_attributes(dataframe)
        self.add_buy_criteria(dataframe)

        if self.should_check_for_alarm(dataframe, pair):
            ongoing_df = self.build_ongoing_dataframe(dataframe, pair)
            if self.should_run_alarm(ongoing_df):
                self.alarm_emitted[pair] = True
                msg = f"Spring en {metadata['pair']} en {self.timeframe} con fecha {(datetime.utcnow() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')}"
                print(msg)
                os.system(
                    f"notify-send \"{msg.upper()}\" --urgency critical -i /usr/share/icons/gnome/48x48/actions/stock_about.png")

        return dataframe

    def add_volume_condition(self, dataframe):
        def highest(df, window):
            return df.rolling(window).max()

        def ema(df, length):
            return ta.EMA(df, length)

        volume = dataframe["volume"]
        volmax = highest(volume, 89)
        vol = volume * 100 / volmax * 4 / 5
        volpmed = ema(vol, 21)
        hvpm = vol - volpmed
        niv_crit = highest(hvpm, 89) * 0.618

        dataframe["volume_condition"] = np.where(
            (hvpm > 0) & (hvpm < niv_crit), 0, np.where(
                (hvpm > 0) & (hvpm >= niv_crit) & (dataframe["close"] < dataframe["open"]), 1, 0))

    def add_volume_condition_attributes(self, dataframe):
        dataframe["volume_condition_close"] = np.where(
            dataframe["volume_condition"] == 1,
            dataframe["close"],
            np.nan
        )
        dataframe["volume_condition_close"].ffill(inplace=True)

        dataframe["volume_condition_open"] = np.where(
            dataframe["volume_condition"] == 1,
            dataframe["open"],
            np.nan
        )
        dataframe["volume_condition_open"].ffill(inplace=True)

        dataframe["volume_condition_off_range"] = np.where(
            dataframe["volume_condition"] == 1,
            0,
            np.where(
                (dataframe["close"] >
                 dataframe["volume_condition_open"] * self.volume_condition_off_range_top_limit_multiplier) |
                (dataframe["close"] <
                 dataframe["volume_condition_close"] * self.volume_condition_off_range_bottom_limit_multiplier),
                1,
                np.nan
            )
        )
        dataframe["volume_condition_off_range"].ffill(inplace=True)

        macd = ta.MACD(dataframe)
        dataframe["macdhist"] = macd["macdhist"]
        dataframe["volume_condition_macdhist"] = np.where(
            dataframe["volume_condition"] == 1,
            dataframe["macdhist"],
            np.nan
        )
        dataframe["volume_condition_macdhist"].ffill(inplace=True)

        dataframe["volume_condition_recession"] = np.where(
            dataframe["volume_condition"] == 1,
            0,
            np.where(
                dataframe["close"] > (
                        ((dataframe["volume_condition_open"] - dataframe["volume_condition_close"]) / 2) +
                        dataframe["volume_condition_close"]
                ),
                1,
                np.nan
            )
        )
        dataframe["volume_condition_recession"].ffill(inplace=True)

    def add_buy_criteria(self, dataframe):
        dataframe["buy_criteria"] = (
                (dataframe["macdhist"].shift(2) < 0) &
                (dataframe["macdhist"].shift(1) < 0) &
                (dataframe["macdhist"].shift(1) < dataframe["macdhist"].shift(2)) &
                (dataframe["macdhist"].shift(1) < dataframe["macdhist"]) &
                (dataframe["close"].shift(1) < dataframe["volume_condition_close"]) &
                (dataframe["volume_condition_macdhist"] < dataframe["macdhist"].shift(1)) &
                (dataframe["volume_condition_recession"] == 1) &
                (dataframe["volume_condition_off_range"] == 0)
        )

    def should_check_for_alarm(self, dataframe, pair):
        return datetime.now().minute >= self.start_alarm_check_minute and \
               not self.alarm_emitted[pair] and \
               dataframe["macdhist"].shift(1).iloc[-1] < 0 and \
               dataframe["macdhist"].iloc[-1] < 0 and \
               dataframe["macdhist"].iloc[-1] < dataframe["macdhist"].shift(1).iloc[-1] and \
               dataframe["close"].iloc[-1] < dataframe["volume_condition_close"].iloc[-1] and \
               dataframe["volume_condition_macdhist"].iloc[-1] < dataframe["macdhist"].iloc[-1] and \
               dataframe["volume_condition_recession"].iloc[-1] == 1 and \
               dataframe["volume_condition_off_range"].iloc[-1] == 0 and \
               self.dp and \
               self.dp.runmode.value in ('live', 'dry_run')

    def build_ongoing_dataframe(self, dataframe, pair):
        ticker = self.dp.ticker(pair)
        ongoing_df = dataframe.append(Series({
            'volume': 0,  # 0 volume for the on-going candle, does not affect the alarm
            'open': ticker['open'],
            'high': ticker['high'],
            'low': ticker['low'],
            'close': ticker['close'],
            'volume_condition_close': dataframe['volume_condition_close'].iloc[-1]
        }), ignore_index=True)

        macd = ta.MACD(ongoing_df)
        ongoing_df['macdhist'] = macd["macdhist"]

        return ongoing_df

    def should_run_alarm(self, ongoing_df):
        return ongoing_df["macdhist"].shift(1).iloc[-1] < ongoing_df["macdhist"].iloc[-1]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["buy_criteria"]
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
