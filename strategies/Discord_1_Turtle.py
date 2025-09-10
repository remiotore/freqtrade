# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta

#=========================================
# Using WinLoss HyperOptFunction :
#   # Buy hyperspace params:
#     buy_params = {
#      'rmax-value': 1012
#     }

#     # Sell hyperspace params:
#     sell_params = {
#      'rmin-value': 612
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.35182,
#         "86": 0.11621,
#         "128": 0.023,
#         "458": 0
#     }

#     # Stoploss:
#     stoploss = -0.32048
#=======================================
#
# # Using SHarpe HyperOptFunction :
# # Buy hyperspace params:
#     buy_params = {
#      'rmax-value': 520
#     }

#     # Sell hyperspace params:
#     sell_params = {
#      'rmin-value': 75
#     }

#     # ROI table:
#     minimal_roi = {
#         "0": 0.31565,
#         "114": 0.15022,
#         "224": 0.04378,
#         "405": 0
#     }

#     # Stoploss:
#     stoploss = -0.18586
#==========================================


class Turtle(IStrategy):

    # ROI table:
    minimal_roi = {
        "0": 0.31565,
        "114": 0.15022,
        "224": 0.04378,
        "405": 0
    }

    # Stoploss:
    stoploss = -0.18586

    startup_candle_count = 520

    ticker_interval = '15m'
    timeframe = '15m'

    # trailing_stop = True
    # trailing_stop_positive = 0.01
    # trailing_only_offset_is_reached = True
    # trailing_stop_positive_offset = 0.1

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.01116
    trailing_only_offset_is_reached = True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rmax'] = dataframe['close'].rolling(520).max()
        dataframe['rmin'] = dataframe['close'].rolling(75).min()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['high'] >= dataframe['rmax'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['low'] <= dataframe['rmin'])
            ),
            'sell'] = 1
        return dataframe
