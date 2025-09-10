
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import os

import numpy as np
import talib.abstract as ta

from freqtrade.rpc import RPCMessageType
from freqtrade.utils.trades_manager import TradeManager

from freqtrade.utils.tradingview import generate_tv_url
from freqtrade.utils.binance_rest_api import get_ongoing_candle

from typing import List, Tuple, Dict

import logging

from colorama import Fore, Style

logger = logging.getLogger(__name__)


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def calculate_percentage_change(start_value: float, final_value: float) -> float:
    if final_value == 0:
        return 0
    return (final_value - start_value) / start_value * 100


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


def yellow_text(text):
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


class RSIDropDNSTraderBTCVWAP(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '5m'



    alarm_emitted = dict()
    notifications = dict()
    max_bars_back = 500
    max_simultaneous_engulf_patterns = 10
    BTC_ETH = ["BTC", "ETH"]

    def __init__(self, config: dict) -> None:
        self.btc_eth_alert_percentage = float(config['btc_eth_alert_percentage'])
        self.altcoins_alert_percentage = float(config['altcoins_alert_percentage'])
        self.btc_eth_restart_alert_percentage = float(config['btc_eth_restart_alert_percentage'])
        self.altcoins_restart_alert_percentage = float(config['altcoins_restart_alert_percentage'])
        self.profit_rate = 2
        self.trade_manager_by_pair: Dict[str, TradeManager] = dict()
        super().__init__(config)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if pair not in self.trade_manager_by_pair:
            self.trade_manager_by_pair[pair] = TradeManager(profit_rate=self.profit_rate)
            self.notifications[pair] = {
                "notify_buy": False,
                "notify_stop_loss": False,
                "notify_profit": False,
            }
        short_df = dataframe.tail(self.max_bars_back)

        tv_interval = 5
        if self.notifications[pair]['notify_buy']:
            msg = f"{pair} bought {self.timeframe} {datetime.now()}"
            os.system(
                f"notify-send \"{msg}\" --urgency critical -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            print(yellow_text(
                f"https://www.tradingview.com/chart/?symbol=binance:{pair.replace('/', '')}&interval={tv_interval}"))
            self.notifications[pair]['notify_buy'] = False
        if self.notifications[pair]['notify_stop_loss']:
            msg = f"{pair} stop loss run {self.timeframe} {datetime.now()}"
            os.system(
                f"notify-send \"{msg}\" --urgency critical -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            print(yellow_text(
                f"https://www.tradingview.com/chart/?symbol=binance:{pair.replace('/', '')}&interval={tv_interval}"))
            self.notifications[pair]['notify_stop_loss'] = False
        if self.notifications[pair]['notify_profit']:
            msg = f"{pair} profit reached {self.timeframe} {datetime.now()}"
            os.system(
                f"notify-send \"{msg}\" --urgency critical -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            print(yellow_text(
                f"https://www.tradingview.com/chart/?symbol=binance:{pair.replace('/', '')}&interval={tv_interval}"))
            self.notifications[pair]['notify_profit'] = False

        if self.dp and \
                self.dp.runmode.value in ('live', 'dry_run'):
            short_df = short_df.append(get_ongoing_candle(pair=pair, timeframe=self.timeframe), ignore_index=True)

        previous_range = short_df["open"].shift(1) - short_df["close"].shift(1)

        bull_rsi_threshold = 20  # 25

        short_df["bull_engulf_red_line"] = self.calculate_bull_engulf_red_line(dataframe=short_df)

        short_df["bull_engulf_green_line"] = self.calculate_bull_engulf_green_line(
            previous_range=previous_range, dataframe=short_df, bull_engulf_red_line=short_df["bull_engulf_red_line"],
            rsi_threshold=bull_rsi_threshold)



        ongoing_close = short_df['close'].iloc[-1]
        bull_engulf_green_line_list = short_df["bull_engulf_green_line"].dropna().tail(
            self.max_simultaneous_engulf_patterns).tolist()



        buy_criteria, sell_criteria = False, False
        closest_bull_green_line, closest_bull_red_line = self.get_closest_bull_green_red_line(dataframe=short_df)
        trade_manager = self.trade_manager_by_pair[pair]

        drop_rsi_threshold = 35

        if not trade_manager.has_open_trade:
            if self.should_buy(pair=pair, dataframe=short_df, ongoing_close=ongoing_close,
                               closest_bull_green_line=closest_bull_green_line,
                               closest_bull_red_line=closest_bull_red_line,
                               drop_rsi_threshold=drop_rsi_threshold):
                buy_criteria = True
                self.notifications[pair]['notify_buy'] = True
                trade_manager.open_trade(buy_price=ongoing_close, stop_loss_price=closest_bull_red_line)
                print(yellow_text(
                    f"buy: {trade_manager.buy_price} stoploss: {trade_manager.stop_loss_price} profit: {trade_manager.profit_price}"))
        else:
            if trade_manager.should_stop_loss(ongoing_close=ongoing_close):
                sell_criteria = True
                self.notifications[pair]['notify_stop_loss'] = True
                trade_manager.close_trade()
            elif trade_manager.should_profit(ongoing_close=ongoing_close):
                sell_criteria = True
                self.notifications[pair]['notify_profit'] = True
                trade_manager.close_trade()

        dataframe["buy_criteria"] = buy_criteria
        dataframe["sell_criteria"] = sell_criteria

        return dataframe

    def calculate_bull_engulf_red_line(self, dataframe: DataFrame) -> Series:
        return dataframe["low"].rolling(4).min()



    def calculate_bull_engulf_green_line(self, bull_engulf_red_line: Series, previous_range: Series,
                                         dataframe: DataFrame, rsi_threshold: float) -> Series:
        open = dataframe["open"]
        low = dataframe["low"]
        close = dataframe["close"]

        is_bull_engulf = (
                (previous_range > 0) &
                (close > open.shift(1))
        )

        low_list = low.tolist()
        min_low_to_end = []
        for i in range(0, len(low_list)):
            min_low_to_end.append(min(low_list[i:]))
        dataframe["min_low_to_end"] = min_low_to_end

        rsi = ta.RSI(dataframe, timeperiod=14).tolist()
        next_4_candles_rsi_change = [0.0] * len(rsi)
        for i in range(0, len(rsi) - 4):
            next_4_candles_rsi_change[i] = calculate_percentage_change(
                start_value=rsi[i], final_value=rsi[i + 4]
            )
        dataframe["next_4_candles_rsi_change"] = next_4_candles_rsi_change

        return np.where(
            is_bull_engulf &
            (dataframe["min_low_to_end"] >= bull_engulf_red_line) &
            (dataframe["next_4_candles_rsi_change"].shift(1).abs() >= rsi_threshold),
            open.shift(1),
            np.nan
        )

    def calculate_bear_engulf_red_line(self, dataframe: DataFrame) -> Series:
        high = dataframe["high"]
        return np.where(high > high.shift(1), high, high.shift(1))

    def calculate_bear_engulf_green_line(self, bear_engulf_red_line: Series, previous_range: Series,
                                         dataframe: DataFrame, rsi_threshold: float) -> Series:
        open = dataframe["open"]
        high = dataframe["high"]
        close = dataframe["close"]

        is_bear_engulf = (
                (previous_range < 0) &
                (close < open.shift(1))
        )

        high_list = high.tolist()
        max_high_to_end = []
        for i in range(0, len(high_list)):
            max_high_to_end.append(max(high_list[i:]))
        dataframe["max_high_to_end"] = max_high_to_end

        return np.where(
            is_bear_engulf &
            (dataframe["max_high_to_end"] <= bear_engulf_red_line),
            open.shift(1),
            np.nan
        )

    def rsi_in_range(self, pair, dataframe, rsi_threshold):
        rsi = ta.RSI(dataframe, timeperiod=14).tolist()
        lookback_candles = 7
        last_rsi = rsi[-1]
        result = False
        for i in range(2, lookback_candles + 1):
            if calculate_percentage_change(last_rsi, rsi[-i]) > rsi_threshold:

                result = True
                break
        return result

    def get_closest_bull_green_red_line(self, dataframe: DataFrame) -> Tuple[float, float] or Tuple[None, None]:
        filter_df = dataframe[dataframe["bull_engulf_green_line"].notnull()]
        if not filter_df.empty:
            closest_bull_green_line = filter_df["bull_engulf_green_line"].iloc[-1]
            closest_bull_red_line = filter_df["bull_engulf_red_line"].iloc[-1]
            return closest_bull_green_line, closest_bull_red_line
        return None, None

    def should_buy(self, pair, dataframe: DataFrame, ongoing_close: float,
                   closest_bull_green_line: float, closest_bull_red_line: float, drop_rsi_threshold: float) -> bool:
        if closest_bull_green_line and closest_bull_red_line:
            return self.rsi_in_range(pair=pair, dataframe=dataframe, rsi_threshold=drop_rsi_threshold) and \
                   closest_bull_green_line >= ongoing_close >= closest_bull_red_line
        return False

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["buy_criteria"]
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["sell_criteria"]
            ),
            'sell'] = 1
        return dataframe
