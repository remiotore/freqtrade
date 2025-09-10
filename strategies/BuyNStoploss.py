
import os

from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta



class BuyNStoploss(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '1h'

    bought_once = False
    stop_loss_once = False
    notify_buy = False
    notify_stop_loss = False

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

        buy = False
        sell = False

        current_price = self.dp.ticker(pair)["last"]
        if not self.bought_once and self.buy_zone_price_top >= current_price > self.buy_zone_price_bottom:
            self.bought_once = True
            self.notify_buy = True
            buy = True
        if not self.stop_loss_once and current_price < self.buy_zone_price_bottom:
            self.stop_loss_once = True
            self.notify_stop_loss = True
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
