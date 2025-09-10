
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import os

import numpy as np

from freqtrade.rpc import RPCMessageType
from beepy import beep

from technical.util import resample_to_interval

from typing import List

from colorama import Fore, Style

from pandas import to_datetime

from freqtrade.utils.binance_rest_api import get_candles, get_ongoing_candle


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


def green(text):
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


def red(text):
    return f"{Fore.RED}{text}{Style.RESET_ALL}"


def get_now_and_dataframe_hour_diff(dataframe):
    last_candle_closed_date = dataframe["date"].iloc[-1].replace(tzinfo=None)
    now = datetime.utcnow().replace(tzinfo=None)

    return (now - last_candle_closed_date).total_seconds() / 3600


class DNSAlarmReporter(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '30m'



    alarm_emitted = dict()
    max_bars_back = 500
    max_simultaneous_engulf_patterns = 10
    BTC_ETH = ["BTC", "ETH"]

    df_dict = dict()

    def __init__(self, config: dict) -> None:
        self.btc_eth_alert_percentage = float(config['btc_eth_alert_percentage'])
        self.altcoins_alert_percentage = float(config['altcoins_alert_percentage'])
        self.btc_eth_restart_alert_percentage = float(config['btc_eth_restart_alert_percentage'])
        self.altcoins_restart_alert_percentage = float(config['altcoins_restart_alert_percentage'])
        super().__init__(config)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        df_30m = dataframe
        if self.dp and \
                self.dp.runmode.value in ('live', 'dry_run'):
            df_30m = df_30m.append(get_ongoing_candle(pair=pair, timeframe=self.timeframe), ignore_index=True)
        df_1h = resample_to_interval(df_30m, 60)

        if pair not in self.df_dict:
            self.df_dict[pair] = None
            self.df_dict[f"{pair}2h"] = None
            self.df_dict[f"{pair}4h"] = None
            self.df_dict[f"{pair}1d"] = None
            self.df_dict[f"{pair}1w"] = None

        df_2h = self.df_dict[f"{pair}2h"]
        df_4h = self.df_dict[f"{pair}4h"]
        df_1d = self.df_dict[f"{pair}1d"]
        df_1w = self.df_dict[f"{pair}1w"]
        if self.dp and \
                self.dp.runmode.value in ('live', 'dry_run'):
            if df_2h is None or get_now_and_dataframe_hour_diff(df_2h) > 2:
                self.df_dict[f"{pair}2h"] = get_candles(pair=pair, timeframe="2h")
                df_2h = self.df_dict[f"{pair}2h"]
            if df_4h is None or get_now_and_dataframe_hour_diff(df_4h) > 4:
                self.df_dict[f"{pair}4h"] = get_candles(pair=pair, timeframe="4h")
                df_4h = self.df_dict[f"{pair}4h"]
            if df_1d is None:
                self.df_dict[f"{pair}1d"] = get_candles(pair=pair, timeframe="1d")
                df_1d = self.df_dict[f"{pair}1d"]
            if df_1w is None:
                self.df_dict[f"{pair}1w"] = get_candles(pair=pair, timeframe="1w")
                df_1w = self.df_dict[f"{pair}1w"]

        ongoing_close = df_30m["close"].iloc[-1]

        self.calculate_dns(df_30m, ongoing_close, pair, "30m")

        self.calculate_dns(df_1h, ongoing_close, pair, "1h")
        self.calculate_dns(df_2h, ongoing_close, pair, "2h")
        self.calculate_dns(df_4h, ongoing_close, pair, "4h")
        self.calculate_dns(df_1d, ongoing_close, pair, "1d")
        self.calculate_dns(df_1w, ongoing_close, pair, "1w")
        print("")

        return dataframe

    def calculate_dns(self, dataframe, ongoing_close, pair, timeframe):
        short_df = dataframe.tail(self.max_bars_back)







        previous_range = short_df["open"].shift(1) - short_df["close"].shift(1)

        short_df["bull_engulf_green_line"] = self.calculate_bull_engulf_green_line(
            previous_range=previous_range, dataframe=short_df)
        short_df["bear_engulf_green_line"] = self.calculate_bear_engulf_green_line(
            previous_range=previous_range, dataframe=short_df)

        short_df["bull_engulf_red_line"] = self.calculate_bull_engulf_red_line(
            previous_range=previous_range, dataframe=short_df)
        short_df["bear_engulf_red_line"] = self.calculate_bear_engulf_red_line(
            previous_range=previous_range, dataframe=short_df)







        bull_engulf_green_line_list = short_df["bull_engulf_green_line"].dropna().tail(
            self.max_simultaneous_engulf_patterns).tolist()
        bear_engulf_green_line_list = short_df["bear_engulf_green_line"].dropna().tail(
            self.max_simultaneous_engulf_patterns).tolist()

        bull_engulf_red_line_list = short_df["bull_engulf_red_line"].dropna().tail(
            self.max_simultaneous_engulf_patterns).tolist()
        bear_engulf_red_line_list = short_df["bear_engulf_red_line"].dropna().tail(
            self.max_simultaneous_engulf_patterns).tolist()



        def get_closest_and_smaller(n: float, l: List[float]):
            result = None
            for v in l:
                if v < n:
                    if result is None or (n - v) < (n - result):
                        result = v
            return result

        def get_closest_and_greater(n: float, l: List[float]):
            result = None
            for v in l:
                if v > n:
                    if result is None or (v - n) < (result - n):
                        result = v
            return result

        closest_demand_green_line = get_closest_and_smaller(ongoing_close, bull_engulf_green_line_list)
        closest_demand_red_line = get_closest_and_smaller(ongoing_close, bull_engulf_red_line_list)
        closest_offer_green_line = get_closest_and_greater(ongoing_close, bear_engulf_green_line_list)
        closest_offer_red_line = get_closest_and_greater(ongoing_close, bear_engulf_red_line_list)

        if closest_demand_green_line:
            distance_closest_demand_green_line = \
                round(calculate_distance_percentage(ongoing_close, closest_demand_green_line), 2)
        else:
            distance_closest_demand_green_line = "-"

        if closest_demand_red_line:
            distance_closest_demand_red_line = \
                round(calculate_distance_percentage(ongoing_close, closest_demand_red_line), 2)
        else:
            distance_closest_demand_red_line = "-"

        if closest_offer_green_line:
            distance_closest_offer_green_line = \
                round(calculate_distance_percentage(ongoing_close, closest_offer_green_line), 2)
        else:
            distance_closest_offer_green_line = "-"

        if closest_offer_red_line:
            distance_closest_offer_red_line = \
                round(calculate_distance_percentage(ongoing_close, closest_offer_red_line), 2)
        else:
            distance_closest_offer_red_line = "-"

        def any_under_threshold(pair_: str, *distance_percentages):
            result = False
            for d in distance_percentages:
                if d == "-":
                    continue
                if get_symbol_from_pair(pair_).upper() in self.BTC_ETH:
                    if d < self.btc_eth_alert_percentage:
                        result = True
                else:
                    if d < self.altcoins_alert_percentage:
                        result = True
            return result

        desktop_notif_text = f'{pair} {timeframe} BUY: {distance_closest_demand_green_line} {distance_closest_demand_red_line} ' \
                             f'SELL: {distance_closest_offer_green_line} {distance_closest_offer_red_line}'
        text = f'{pair} {timeframe} BUY: {green(distance_closest_demand_green_line)} {red(distance_closest_demand_red_line)} ' \
               f'SELL: {green(distance_closest_offer_green_line)} {red(distance_closest_offer_red_line)}'
        if any_under_threshold(pair, distance_closest_demand_green_line, distance_closest_demand_red_line):

            if os.getenv("beep") == "beep":

                os.system(f"notify-send \"{desktop_notif_text.upper()}\" -t 10000 -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
                
            text += "    BUY    "
        if any_under_threshold(pair, distance_closest_offer_green_line, distance_closest_offer_red_line):
            text += "    SELL    "
        print(text)
























    def get_ongoing_candle(self, pair: str) -> Series:
        ticker = self.dp.ticker(pair)
        ongoing_open = ticker['open']
        ongoing_high = ticker['high']
        ongoing_low = ticker['low']
        ongoing_close = ticker['close']
        return Series({
            'volume': 0,  # 0 volume for the on-going candle, does not affect the alarm
            'open': ongoing_open,
            'high': ongoing_high,
            'low': ongoing_low,
            'close': ongoing_close
        })

    def calculate_bull_engulf_green_line(self, previous_range: Series, dataframe: DataFrame) -> Series:
        open = dataframe["open"]
        low = dataframe["low"]
        close = dataframe["close"]

        is_bull_engulf = (
                (previous_range > 0) &
                (close > open.shift(1))
        )

        bull_engulf_low = np.where(low < low.shift(1), low, low.shift(1))

        low_list = low.tolist()
        min_low_to_end = []
        for i in range(0, len(low_list)):
            min_low_to_end.append(min(low_list[i:]))
        dataframe["min_low_to_end"] = min_low_to_end

        return np.where(
            is_bull_engulf &
            (dataframe["min_low_to_end"] >= bull_engulf_low),
            open.shift(1),
            np.nan
        )

    def calculate_bear_engulf_green_line(self, previous_range: Series, dataframe: DataFrame) -> Series:
        open = dataframe["open"]
        high = dataframe["high"]
        close = dataframe["close"]

        is_bear_engulf = (
                (previous_range < 0) &
                (close < open.shift(1))
        )

        bear_engulf_high = np.where(high > high.shift(1), high, high.shift(1))

        high_list = high.tolist()
        max_high_to_end = []
        for i in range(0, len(high_list)):
            max_high_to_end.append(max(high_list[i:]))
        dataframe["max_high_to_end"] = max_high_to_end

        return np.where(
            is_bear_engulf &
            (dataframe["max_high_to_end"] <= bear_engulf_high),
            open.shift(1),
            np.nan
        )

    def calculate_bull_engulf_red_line(self, previous_range: Series, dataframe: DataFrame) -> Series:
        open = dataframe["open"]
        low = dataframe["low"]
        close = dataframe["close"]

        is_bull_engulf = (
                (previous_range > 0) &
                (close > open.shift(1))
        )

        bull_engulf_low = np.where(low < low.shift(1), low, low.shift(1))

        low_list = low.tolist()
        min_low_to_end = []
        for i in range(0, len(low_list)):
            min_low_to_end.append(min(low_list[i:]))
        dataframe["min_low_to_end"] = min_low_to_end

        return np.where(
            is_bull_engulf &
            (dataframe["min_low_to_end"] >= bull_engulf_low),
            bull_engulf_low,
            np.nan
        )

    def calculate_bear_engulf_red_line(self, previous_range: Series, dataframe: DataFrame) -> Series:
        open = dataframe["open"]
        high = dataframe["high"]
        close = dataframe["close"]

        is_bear_engulf = (
                (previous_range < 0) &
                (close < open.shift(1))
        )

        bear_engulf_high = np.where(high > high.shift(1), high, high.shift(1))

        high_list = high.tolist()
        max_high_to_end = []
        for i in range(0, len(high_list)):
            max_high_to_end.append(max(high_list[i:]))
        dataframe["max_high_to_end"] = max_high_to_end

        return np.where(
            is_bear_engulf &
            (dataframe["max_high_to_end"] <= bear_engulf_high),
            bear_engulf_high,
            np.nan
        )

    def add_backtest_missing_candles(self, dataframe: DataFrame):
        from datetime import datetime
        import pytz
        utc = pytz.UTC



        dataframe.append(
            {"date": utc.localize(datetime(year=2021, month=5, day=31, minute=0, second=0, microsecond=0)),
             "open": 0,
             "high": 0,
             "low": 0,
             "close": 0,
             "volume": 0}, ignore_index=True)

    def is_price_in_alert_range(self, pair: str, distance_percentage: float) -> bool:
        if get_symbol_from_pair(pair).upper() in self.BTC_ETH:
            return distance_percentage < self.btc_eth_alert_percentage
        return distance_percentage < self.altcoins_alert_percentage

    def is_price_in_restart_alert_range(self, pair: str, distance_percentage: float) -> bool:
        if get_symbol_from_pair(pair).upper() in self.BTC_ETH:
            return distance_percentage > self.btc_eth_restart_alert_percentage
        return distance_percentage > self.altcoins_restart_alert_percentage

    def build_alert_message(self, pair: str, green_line_price: float) -> str:
        if get_symbol_from_pair(pair).upper() in self.BTC_ETH:
            alert_percentage = self.btc_eth_alert_percentage
        else:
            alert_percentage = self.altcoins_alert_percentage
        return f"{pair} se encuentra a menos de {round(alert_percentage, 2)}% " \
               f"de {round(green_line_price, 2)} con fecha " \
               f"{(datetime.utcnow() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')} ARG"

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
