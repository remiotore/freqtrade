from freqtrade.strategy.interface import IStrategy
from pandas.core.frame import DataFrame
from functools import reduce
from mcDuck.custom_indicators import simpleTrendReversal


class StrategyTrendReversal(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '5m'
    timeframe_support = '5m'
    timeframe_main = '5m'

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    ignore_buying_expired_candle_after = 5

    startup_candle_count: int = 4

    minimal_roi = {
        "0": 0.179,
        "26": 0.093,
        "80": 0.035,
        "177": 0
    }

    stoploss = -0.316

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = simpleTrendReversal(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        minimum_coin_price = 0.0000015
        conditions = []
        conditions.append(dataframe["volume"] > 0)
        conditions.append(dataframe["close"] > minimum_coin_price)
        conditions.append(dataframe["trend_reversal"] == True)
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "buy"] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_sell_trend(dataframe, metadata)
