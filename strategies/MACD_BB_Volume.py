from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MACD_BB_Volume(IStrategy):
    """
    MACD with Bollinger Bands and Volume Filter Strategy
    """

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.05  # 5% ROI
    }

    # Stoploss:
    stoploss = -0.015  # -1.5%

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.008  # 0.8%
    trailing_stop_positive_offset = 0.01  # 1%
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '1h'

    # Define the indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['signal']
        dataframe['macd_diff'] = macd['macd'] - macd['signal']

        # Bollinger Bands
        bollinger = ta.BBAND(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']

        # Volume Filter (Volume SMA)
        dataframe['volume_sma'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['volume_above_sma'] = dataframe['volume'] > dataframe['volume_sma']

        return dataframe

    # Generate buy signals
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['macd'] > dataframe['macd_signal']) &  # MACD bullish crossover
                (dataframe['close'] <= dataframe['bb_lower']) &    # Price touches or below lower Bollinger Band
                (dataframe['volume_above_sma'])                    # Volume above SMA
            ),
            'buy'] = 1
        return dataframe

    # Generate sell signals
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['macd'] < dataframe['macd_signal']) &  # MACD bearish crossover
                (dataframe['close'] >= dataframe['bb_upper']) &    # Price touches or above upper Bollinger Band
                (dataframe['volume_above_sma'])                    # Volume above SMA
            ),
            'sell'] = 1
        return dataframe