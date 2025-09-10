from freqtrade.strategy.interface import IStrategy


class strategy_3(IStrategy):
    INTERFACE_VERSION = 2
    minimal_roi = {"0": 9999}
    stoploss = -0.99
    trailing_stop = False




    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True


    startup_candle_count: int = 800

    order_types = {
        "buy": "limit",
        "sell": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"buy": "gtc", "sell": "gtc"}
