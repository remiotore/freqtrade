from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib

# в „в „в „вў°вЈ§вЈјвЈҜв „вЈёвЈ вЈ¶вЈ¶вЈҰвЈҫв „в „в „в „вЎҖв „вўҖвЈҝвЈҝв „в „в „вўёвЎҮв „в „
# в „в „в „вЈҫвЈҝв ҝв ҝв ¶в ҝвўҝвЈҝвЈҝвЈҝвЈҝвЈҰвЈӨвЈ„вўҖвЎ…вў вЈҫвЈӣвЎүв „в „в „в ёвўҖвЈҝв „
# в „в „вўҖвЎӢвЈЎвЈҙвЈ¶вЈ¶вЎҖв „в „в ҷвўҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҙвЈҝвЈҝвЈҝвўғвЈӨвЈ„вЈҖвЈҘвЈҝвЈҝв „
# в „в „вўёвЈҮв »вЈҝвЈҝвЈҝвЈ§вЈҖвўҖвЈ вЎҢвў»вЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝв ҝв ҝв ҝвЈҝвЈҝвЈҝв „
# в „вўҖвўёвЈҝвЈ·вЈӨвЈӨвЈӨвЈ¬вЈҷвЈӣвўҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЎҝвЈҝвЈҝвЎҚв „в „вўҖвЈӨвЈ„в үв ӢвЈ°
# в „вЈјвЈ–вЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвўҝвЈҝвЈҝвЈҝвЈҝвЈҝвўҮвЈҝвЈҝвЎ·в ¶в ¶вўҝвЈҝвЈҝв ҮвўҖвЈӨ
# в ҳвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҪвЈҝвЈҝвЈҝвЎҮвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈ·вЈ¶вЈҘвЈҙвЈҝвЎ—
# вўҖв ҲвўҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЎҹв „
# вўёвЈҝвЈҰвЈҢвЈӣвЈ»вЈҝвЈҝвЈ§в ҷв ӣв ӣвЎӯв …в ’в Ұв ӯвЈӯвЎ»вЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЎҝв ғв „
# в ҳвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЎҶв „в „в „в „в „в „в „в „в №в ҲвўӢвЈҪвЈҝвЈҝвЈҝвЈҝвЈөвЈҫв ғв „
# в „в ҳвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝв „вЈҙвЈҝвЈ¶вЈ„в „вЈҙвЈ¶в „вўҖвЈҫвЈҝвЈҝвЈҝвЈҝвЈҝвЈҝв ғв „в „
# в „в „в Ҳв »вЈҝвЈҝвЈҝвЈҝвЈҝвЈҝвЎ„вў»вЈҝвЈҝвЈҝв „вЈҝвЈҝвЎҖвЈҫвЈҝвЈҝвЈҝвЈҝвЈӣв ӣв Ғв „в „в „
# в „в „в „в „в Ҳв ӣвўҝвЈҝвЈҝвЈҝв Ғв һвўҝвЈҝвЈҝвЎ„вўҝвЈҝвЎҮвЈёвЈҝвЈҝв ҝв ӣв Ғв „в „в „в „в „
# в „в „в „в „в „в „в „в үв »вЈҝвЈҝвЈҫвЈҰвЎҷв »вЈ·вЈҫвЈҝв ғв ҝв Ӣв Ғв „в „в „в „в „вўҖвЈ вЈҙ
# вЈҝвЈҝвЈҝвЈ¶вЈ¶вЈ®вЈҘвЈ’в Івў®вЈқвЎҝвЈҝвЈҝвЎҶвЈҝвЎҝв ғв „в „в „в „в „в „в „вЈ вЈҙвЈҝвЈҝвЈҝ


DUALFIT = False
COUNT = 10
GAP = 3



class Precognition(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_fast": 2,
        "buy_push": 1.022,
        "buy_shift": -8,
        "buy_slow": 16,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_fast": 34,
        "sell_push": 0.458,
        "sell_shift": -8,
        "sell_slow": 44,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.166,
        "44": 0.012,
        "59": 0
    }

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'emergencysell': 'limit',
        'forcebuy': "limit",
        'forcesell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # Stoploss:
    stoploss = -0.194

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    # Buy hypers
    timeframe = '5m'
    # #################### END OF RESULT PLACE ####################
    buy_push = DecimalParameter(0, 2, decimals=3, default=1, space='buy')
    buy_shift = IntParameter(-10, 0, default=-6, space='buy')
    buy_fast = IntParameter(2, 50, default=9, space='buy')
    buy_slow = IntParameter(2, 50, default=18, space='buy')
    if not DUALFIT:
        sell_push = DecimalParameter(
            0, 2, decimals=3,  default=1, space='sell')
        sell_shift = IntParameter(-10, 0, default=-6, space='sell')
        sell_fast = IntParameter(2, 50, default=9, space='sell')
        sell_slow = IntParameter(2, 50, default=18, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['buy_ema_fast'] = ta.SMA(
            dataframe, timeperiod=int(self.buy_fast.value))
        dataframe['buy_ema_slow'] = ta.SMA(
            dataframe, timeperiod=int(self.buy_slow.value))

        conditions = []

        conditions.append(
            qtpylib.crossed_above(
                dataframe['buy_ema_fast'].shift(self.buy_shift.value),
                dataframe['buy_ema_slow'].shift(
                    self.buy_shift.value)*self.buy_push.value
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy']=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        push = self.buy_push.value
        shift = self.buy_shift.value
        ema_fast = dataframe['buy_ema_fast']
        ema_slow = dataframe['buy_ema_slow']

        if not DUALFIT:
            push = self.sell_push.value
            shift = self.sell_shift.value
            ema_fast = dataframe['sell_ema_fast'] = ta.SMA(
                dataframe, timeperiod=int(self.buy_fast.value))
            ema_slow = dataframe['sell_ema_slow'] = ta.SMA(
                dataframe, timeperiod=int(self.buy_slow.value))

        conditions = []

        conditions.append(
            qtpylib.crossed_below(
                ema_fast.shift(shift),
                ema_slow.shift(shift)*push
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell']=1
        return dataframe