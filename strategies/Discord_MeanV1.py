
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
# profitable pairs: 20190601-20201115
# {"exchange": {"pair_whitelist": ["ADA/USDT", "ARPA/USDT", "BAND/USDT", "BLZ/USDT", "BNT/USDT", "BTC/USDT", "CHR/USDT", "CRV/USDT", "CVC/USDT", "DCR/USDT", "DOT/USDT", "HBAR/USDT", "KAVA/USDT", "LINK/USDT", "LRC/USDT", "MANA/USDT", "MATIC/USDT", "OMG/USDT", "REN/USDT", "STORJ/USDT", "VET/USDT", "WAN/USDT", "YFI/USDT"]}}


class MeanV1(IStrategy):
    ticker_interval = '30m'
    startup_candle_count: int = (24*60)/15+20
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    # SET 1 / result from hyperopt
    minimal_roi = {
        "0": 0.11109,
        "3660": 0.08262,
        "10710": 0.07225,
        "25335": 0.04027,
        "38070": 0.03057,
        "48645": 0.02901,
        "61905": 0.02391,
        "64035": 0.02167,
        "70995": 0.01708,
        "76845": 0.00835,
        "80115": 0.00541
    }
    # Stoploss:
    stoploss = -0.08586
    trailing_only_offset_is_reached = False
    trailing_stop = True
    trailing_stop_positive = 0.00101
#    trailing_stop_positive_offset = 0.00359
    # SET 2 / no hyperopt
    minimal_roi = {
        "0": 100
    }
    trailing_stop = False
    stoploss = -0.20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not {'buy', 'sell'}.issubset(dataframe.columns):
            dataframe.loc[:, 'buy'] = 0
            dataframe.loc[:, 'sell'] = 0
        dataframe['typical'] = qtpylib.typical_price(dataframe)
        dataframe['typical_sma'] = qtpylib.sma(dataframe['typical'], window=10)
        min = dataframe['typical'].shift(20).rolling(int(12 * 60 / 15)).min()
        max = dataframe['typical'].shift(20).rolling(int(12 * 60 / 15)).max()
        dataframe['daily_mean'] = (max+min)/2
        return dataframe

    def informative_pairs(self):
        informative_pairs = []
        return informative_pairs

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            qtpylib.crossed_below(
                dataframe['daily_mean'], dataframe['typical_sma'])
        ]
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            qtpylib.crossed_above(
                dataframe['daily_mean'], dataframe['typical_sma'])
        ]
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        return dataframe
