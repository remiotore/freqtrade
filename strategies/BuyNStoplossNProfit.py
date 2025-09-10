
import os

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta



def get_profit_threshold(buy_price: float, stoploss: float, profit_rate: float) -> float:
    binance_total_fee = 0.2
    stoploss_percentage = calculate_distance_percentage(buy_price, stoploss)
    profit_percentage = stoploss_percentage * profit_rate + binance_total_fee
    return buy_price + (buy_price * profit_percentage / 100)


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


class BuyNStoplossNProfit(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '1h'

    bought_once = False
    stop_loss_once = False
    profit_once = False
    notify_buy = False
    notify_stop_loss = False
    notify_profit = False
    open_rate = None

    def __init__(self, config: dict) -> None:
        self.buy_zone_price_top = float(config['buy_zone_price_top'])
        self.buy_zone_price_bottom = float(config['buy_zone_price_bottom'])
        super().__init__(config)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        if self.notify_buy:
            msg = f"{pair} bought"
            os.system(f"notify-send \"{msg}\" --urgency critical -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            self.notify_buy = False
        if self.notify_stop_loss:
            msg = f"{pair} stop loss run :("
            os.system(f"notify-send \"{msg}\" --urgency critical -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            self.notify_stop_loss = False
        if self.notify_profit:
            msg = f"{pair} profit reached!!!"
            os.system(f"notify-send \"{msg}\" --urgency critical -i /usr/share/icons/gnome/48x48/actions/stock_about.png")
            self.notify_stop_loss = False

        buy = False
        sell = False

        current_price = self.dp.ticker(pair)["last"]
        if not self.bought_once and self.buy_zone_price_top >= current_price > self.buy_zone_price_bottom:
            self.bought_once = True
            self.notify_buy = True
            self.open_rate = current_price
            buy = True
        if not self.stop_loss_once and current_price < self.buy_zone_price_bottom:
            self.stop_loss_once = True
            self.notify_stop_loss = True
            sell = True
        
        profit_rate = 1.9
        if not self.profit_once and self.open_rate and current_price >= get_profit_threshold(
                self.open_rate, self.buy_zone_price_bottom, profit_rate):
            self.profit_once = True
            self.notify_profit = True
            sell = True

        dataframe["buy_criteria"] = buy
        dataframe["sell_criteria"] = sell

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["buy_criteria"]
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["sell_criteria"]
            ),
            'sell'] = 1
        return dataframe
