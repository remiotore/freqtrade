from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import IntParameter, DecimalParameter
from freqtrade.persistence import Trade

class FisherTrendStrategy(IStrategy):
    # Hyperoptable parameters
    rsi_length = IntParameter(5, 30, default=14, space="buy", optimize=True)
    ma_length = IntParameter(5, 50, default=20, space="buy", optimize=True)
    fisher_length = IntParameter(2, 40, default=10, space="buy", optimize=True)
    oversold_threshold = DecimalParameter(20, 40, default=30, space="buy", optimize=True)
    overbought_threshold = DecimalParameter(60, 80, default=70, space="sell", optimize=True)

    # Stoploss
    stoploss = -0.4
    stoploss_param = DecimalParameter(-0.15, -0.05, default=-0.1, decimals=3, space="sell", optimize=True)

    # Trailing stop
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04

    minimal_roi = {
        "0": 1000,
        "30": 1000,
        "60": 1000,
    }

    # Leverage
    leverage_value = 5

    # Timeframe
    timeframe = "30m"

    # Informative pairs (e.g., higher timeframe data)
    informative_timeframe = "1h"

    def get_leverage(self, *args, **kwargs) -> float:
        return self.leverage_value

    def leverage(self, pair: str, current_time=None, current_rate=None, current_profit=None, **kwargs) -> float:
        return self.get_leverage()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if 'fish' not in dataframe.columns:
            dataframe['fish'] = 0.0

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_length.value)
        dataframe["rsi_ma"] = ta.SMA(dataframe["rsi"], timeperiod=self.ma_length.value)

        price = (dataframe["high"] + dataframe["low"]) / 2
        max_hl = price.rolling(window=self.fisher_length.value).max()
        min_hl = price.rolling(window=self.fisher_length.value).min()
        value = 2 * ((price - min_hl) / (max_hl - min_hl) - 0.5)

        dataframe["fish"] = 0.33 * value + 0.67 * dataframe["fish"].shift(1)
        dataframe["fish_norm"] = (dataframe["fish"] + 1) * 50

        dataframe["fisher_signal"] = dataframe["fish_norm"]
        dataframe["rsi_signal"] = dataframe["rsi"]
        dataframe["rsi_ma_signal"] = dataframe["rsi_ma"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        oversold = self.oversold_threshold.value
        overbought = self.overbought_threshold.value

        # Long entry
        dataframe.loc[
            (
                (dataframe["rsi"] < oversold) &
                (dataframe["rsi"] > dataframe["rsi_ma"]) &
                (dataframe["fish_norm"] < oversold)
            ),
            "enter_long"
        ] = 1

        # Short entry
        dataframe.loc[
            (
                (dataframe["rsi"] > overbought) &
                (dataframe["rsi"] < dataframe["rsi_ma"]) &
                (dataframe["fish_norm"] > overbought)
            ),
            "enter_short"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        overbought = self.overbought_threshold.value
        oversold = self.oversold_threshold.value

        # Long exit
        dataframe.loc[
            (
                (dataframe["rsi"] > overbought) &
                (dataframe["rsi"] < dataframe["rsi_ma"]) &
                (dataframe["fish_norm"] > overbought)
            ),
            "exit_long"
        ] = 1

        # Short exit
        dataframe.loc[
            (
                (dataframe["rsi"] < oversold) &
                (dataframe["rsi"] > dataframe["rsi_ma"]) &
                (dataframe["fish_norm"] < oversold)
            ),
            "exit_short"
        ] = 1

        return dataframe

    def adjust_trade_position(self, trade: Trade, current_rate: float, current_time) -> float:
        return self.stoploss_param.value
