import numpy as np

from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


class two_ema_rsi_bb(IStrategy):
    """
    author@: idan.shperling
    """
    # ROI table:
    minimal_roi = {
        "0": 0.22025,
        "71": 0.0368,
        "108": 0.01884,
        "453": 0
    }

    # Stoploss:
    stoploss = -0.14706

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01544
    trailing_stop_positive_offset = 0.03901
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '15m'
    #sell_profit_only = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema180'] = ta.EMA(dataframe, timeperiod=180)
        dataframe['ema365'] = ta.EMA(dataframe, timeperiod=365)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband'] = bollinger['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] <= 33) &
                    (dataframe['volume'] > 0) &
                    (dataframe['ema180'] > dataframe['ema365']) &
                    (dataframe['close'] > dataframe['bb_lowerband']) &
                    (dataframe['open'] > dataframe['ema365'])
            ),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] >= 63) |
                    (qtpylib.crossed_above(dataframe['ema365'], dataframe['ema180']))
            ),
            'sell'
        ] = 1

        return dataframe
