
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta

import numpy as np

from freqtrade.rpc import RPCMessageType
from freqtrade.utils.trades_manager import TradeManager

from freqtrade.utils.tradingview import generate_tv_url
from freqtrade.utils.binance_rest_api import get_ongoing_candle

from typing import List, Tuple, Dict

import logging

logger = logging.getLogger(__name__)


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


class DNSAlarm2(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '1w'



    alarm_emitted = dict()
    max_bars_back = 500
    max_simultaneous_engulf_patterns = 10
    BTC_ETH = ["BTC", "ETH"]

    def __init__(self, config: dict) -> None:
        self.btc_eth_alert_percentage = float(config['btc_eth_alert_percentage'])
        self.altcoins_alert_percentage = float(config['altcoins_alert_percentage'])
        self.btc_eth_restart_alert_percentage = float(config['btc_eth_restart_alert_percentage'])
        self.altcoins_restart_alert_percentage = float(config['altcoins_restart_alert_percentage'])
        self.profit_rate = float(config['profit_rate'])
        self.trade_manager_by_pair: Dict[str, TradeManager] = dict()
        super().__init__(config)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        if pair not in self.trade_manager_by_pair:
            self.trade_manager_by_pair[pair] = TradeManager(profit_rate=self.profit_rate)
        short_df = dataframe.tail(self.max_bars_back)

        if self.dp and \
                self.dp.runmode.value in ('live', 'dry_run'):
            short_df = short_df.append(get_ongoing_candle(pair=pair, timeframe=self.timeframe), ignore_index=True)
        elif self.dp.runmode.value.lower() in ["backtest", "plot"]:
            self.add_backtest_missing_candles(dataframe=short_df)

        previous_range = short_df["open"].shift(1) - short_df["close"].shift(1)

        short_df["bull_engulf_red_line"] = self.calculate_bull_engulf_red_line(dataframe=short_df)
        short_df["bear_engulf_red_line"] = self.calculate_bear_engulf_red_line(dataframe=short_df)
        short_df["bull_engulf_green_line"] = self.calculate_bull_engulf_green_line(
            previous_range=previous_range, dataframe=short_df, bull_engulf_red_line=short_df["bull_engulf_red_line"])
        short_df["bear_engulf_green_line"] = self.calculate_bear_engulf_green_line(
            previous_range=previous_range, dataframe=short_df, bear_engulf_red_line=short_df["bear_engulf_red_line"])

        if self.dp.runmode.value.lower() in ["backtest", "plot"]:



            short_df["bull_engulf_green_line"] = short_df["bull_engulf_green_line"].shift(-1)
            short_df["bear_engulf_green_line"] = short_df["bear_engulf_green_line"].shift(-1)

        ongoing_close = short_df['close'].iloc[-1]
        bull_engulf_green_line_list = short_df["bull_engulf_green_line"].dropna().tail(
            self.max_simultaneous_engulf_patterns).tolist()
        bear_engulf_green_line_list = short_df["bear_engulf_green_line"].dropna().tail(
            self.max_simultaneous_engulf_patterns).tolist()

        buy_criteria, sell_criteria = False, False
        closest_bull_green_line, closest_bull_red_line = self.get_closest_bull_green_red_line(dataframe=short_df)
        trade_manager = self.trade_manager_by_pair[pair]
        if not trade_manager.has_open_trade:
            if self.should_buy(ongoing_close=ongoing_close, closest_bull_green_line=closest_bull_green_line,
                               closest_bull_red_line=closest_bull_red_line):
                buy_criteria = True
                trade_manager.open_trade(buy_price=ongoing_close, stop_loss_price=closest_bull_red_line)
        else:
            if trade_manager.should_stop_loss(ongoing_close=ongoing_close) or \
                    trade_manager.should_profit(ongoing_close=ongoing_close):
                sell_criteria = True
                trade_manager.close_trade()

        dataframe["buy_criteria"] = buy_criteria
        dataframe["sell_criteria"] = sell_criteria

        if self.dp.runmode.value in ('live', 'dry_run'):
            self.alarm_bull_green_lines(
                pair=pair, ongoing_close=ongoing_close, bull_engulf_green_line_list=bull_engulf_green_line_list)
            return dataframe
        return short_df

    def alarm_bull_and_bear_green_lines(self, pair: str, ongoing_close: float,
                                        bull_engulf_green_line_list: List[float],
                                        bear_engulf_green_line_list: List[float]):
        lines = bull_engulf_green_line_list + bear_engulf_green_line_list
        self.alarm_lines(pair=pair, ongoing_close=ongoing_close, lines=lines)

    def alarm_bull_green_lines(self, pair: str, ongoing_close: float, bull_engulf_green_line_list: List[float]):
        self.alarm_lines(pair=pair, ongoing_close=ongoing_close, lines=bull_engulf_green_line_list)

    def alarm_lines(self, pair: str, ongoing_close: float, lines: List[float]):
        for line in lines:
            alarm_emitted_key = f"{pair}-{line}"
            if alarm_emitted_key not in self.alarm_emitted:
                self.alarm_emitted[alarm_emitted_key] = False
            distance_percentage = calculate_distance_percentage(
                current_price=ongoing_close, green_line_price=line)
            if self.is_price_in_alert_range(pair=pair, distance_percentage=distance_percentage):
                if not self.alarm_emitted[alarm_emitted_key]:
                    self.alarm_emitted[alarm_emitted_key] = True
                    message = self.build_alert_message(pair=pair, green_line_price=line)
                    self.rpc.send_msg({
                        'type': RPCMessageType.STATUS,
                        'status': message
                    })
            elif self.is_price_in_restart_alert_range(pair=pair, distance_percentage=distance_percentage):
                self.alarm_emitted[alarm_emitted_key] = False

    def calculate_bull_engulf_red_line(self, dataframe: DataFrame) -> Series:
        low = dataframe["low"]
        return np.where(low < low.shift(1), low, low.shift(1))

    def calculate_bull_engulf_green_line(self, bull_engulf_red_line: Series, previous_range: Series,
                                         dataframe: DataFrame) -> Series:
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

        return np.where(
            is_bull_engulf &
            (dataframe["min_low_to_end"] >= bull_engulf_red_line),
            open.shift(1),
            np.nan
        )

    def calculate_bear_engulf_red_line(self, dataframe: DataFrame) -> Series:
        high = dataframe["high"]
        return np.where(high > high.shift(1), high, high.shift(1))

    def calculate_bear_engulf_green_line(self, bear_engulf_red_line: Series, previous_range: Series,
                                         dataframe: DataFrame) -> Series:
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
        tv_section = ""
        try:
            tv_url = generate_tv_url(pair=pair, timeframe=self.timeframe)
            tv_section = f"\nLink a TradingView: {tv_url}"
        except Exception as exception:
            logger.exception(f"Exception in spring alarm: {exception}")
        if get_symbol_from_pair(pair).upper() in self.BTC_ETH:
            alert_percentage = self.btc_eth_alert_percentage
        else:
            alert_percentage = self.altcoins_alert_percentage
        arg_date = (datetime.utcnow() - timedelta(hours=3)).strftime('%d/%m/%Y %H:%M')
        return f"{pair} se encuentra a menos de {round(alert_percentage, 2)}% " \
               f"de {round(green_line_price, 2)} con fecha " \
               f"{arg_date} ARG" \
               f"{tv_section}"

    def get_closest_bull_green_red_line(self, dataframe: DataFrame) -> Tuple[float, float] or Tuple[None, None]:
        filter_df = dataframe[dataframe["bull_engulf_green_line"].notnull()]
        if not filter_df.empty:
            closest_bull_green_line = filter_df["bull_engulf_green_line"].iloc[-1]
            closest_bull_red_line = filter_df["bull_engulf_red_line"].iloc[-1]
            return closest_bull_green_line, closest_bull_red_line
        return None, None

    def should_buy(self, ongoing_close: float, closest_bull_green_line: float, closest_bull_red_line: float) -> bool:
        if closest_bull_green_line and closest_bull_red_line:
            return closest_bull_green_line >= ongoing_close >= closest_bull_red_line
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
