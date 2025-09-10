from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta

class SUI_MeanReversion_Optimized(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    minimal_roi = {"0": 0.10, "60": 0.05, "120": 0.02}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    startup_candle_count: int = 30
    max_open_trades = 2

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                           proposed_stake: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
        last_candle = dataframe.iloc[-1]
        atr = last_candle["atr"]
        wallet = self.wallets.get_available_stake_amount()
        stake = (wallet * 0.01) / (atr / current_rate)
        return min(stake, proposed_stake)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.ema(dataframe["close"], length=20)
        dataframe["ema50"] = ta.ema(dataframe["close"], length=50)
        dataframe["rsi"] = ta.rsi(dataframe["close"], length=14)
        dataframe["atr"] = ta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < 30) &
                (dataframe["close"] < dataframe["ema20"]) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > 70) |
                (dataframe["close"] > dataframe["ema50"])
            ),
            "exit_long"] = 1
        return dataframe