
from freqtrade.strategy import (BooleanParameter, DecimalParameter, IntParameter)
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class main(IStrategy):

    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.05  # Trailing stop loss to arm at 5%
    trailing_stop_positive_offset = 0.03  # Trail by 3%
    trailing_only_offset_is_reached = True
    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True


    buy_params = {
        "fast_ema": IntParameter(5, 20, default=10),
        "slow_ema": IntParameter(30, 100, default=50),
        "rsi_oversold": IntParameter(30, 50, default=40),
        "rsi_overbought": IntParameter(60, 80, default=70),
        "btc_dominance_period": IntParameter(50, 200, default=100)
    }

    def __init__(self) -> None:
        super().__init__()
        self.fast_ema = self.buy_params["fast_ema"]
        self.slow_ema = self.buy_params["slow_ema"]
        self.rsi_oversold = self.buy_params["rsi_oversold"]
        self.rsi_overbought = self.buy_params["rsi_overbought"]
        self.btc_dominance_period = self.buy_params["btc_dominance_period"]

    def analyze(self, dataframe: DataFrame, indicator_list: list) -> DataFrame:

        dataframe['fast_ema'] = ta.EMA(dataframe, timeperiod=self.fast_ema)
        dataframe['slow_ema'] = ta.EMA(dataframe, timeperiod=self.slow_ema)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['btc_dominance'] = ...  # Calculate Bitcoin dominance
        dataframe['btc_dominance_sma'] = dataframe['btc_dominance'].rolling(window=self.btc_dominance_period).mean()

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        buy_condition = (
            (dataframe['fast_ema'] > dataframe['slow_ema']) &
            (dataframe['rsi'] < self.rsi_oversold)
        )

        bullish_sentiment = dataframe['btc_dominance'] < dataframe['btc_dominance_sma']

        dataframe.loc[buy_condition & bullish_sentiment, 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['fast_ema'] < dataframe['slow_ema']) &
            (dataframe['rsi'] > self.rsi_overbought),
            'sell'] = 1

        return DataFrame