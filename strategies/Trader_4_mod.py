from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce


class ImprovedTrader(IStrategy):
    # ROI and Stoploss
    minimal_roi = {
        "0": 1
    }
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 240
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    stoploss = -0.02

    # Custom parameters
    buy_rsi = IntParameter(20, 50, default=30, space="buy", optimize=True)
    buy_cti = DecimalParameter(-1, 1, default=0.5, decimals=2, space="buy", optimize=True)
    sell_fastk = IntParameter(50, 100, default=70, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Basic indicators
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['fastk'], dataframe['fastd'] = ta.STOCHF(dataframe, 5, 3, 3)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['bb_lower'], dataframe['bb_middle'], dataframe['bb_upper'] = ta.BBANDS(dataframe, timeperiod=20)

        # Custom metrics
        dataframe['volatility'] = dataframe['bb_upper'] - dataframe['bb_lower']
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        buy_condition = (
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['cti'] < self.buy_cti.value) &
            (dataframe['close'] < dataframe['sma_15']) &
            (dataframe['volatility'] > dataframe['volatility'].mean()) &
            (dataframe['volume'] > dataframe['volume_mean'])
        )

        conditions.append(buy_condition)
        dataframe.loc[buy_condition, 'enter_tag'] += 'volatility_rsi_cti'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sell_condition = (
            (dataframe['fastk'] > self.sell_fastk.value) |
            (dataframe['close'] > dataframe['sma_50'])  # Exit on trend reversal
        )
        dataframe.loc[sell_condition, ['exit_long', 'exit_tag']] = (1, 'fastk_sma_reversal')
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        # Adaptive stop-loss based on ATR
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty or len(dataframe) < self.startup_candle_count:
            return 1  # Default stoploss

        atr = dataframe['atr'].iloc[-1]
        stoploss = max(-0.03, -2 * atr / current_rate)  # Max loss capped at 3%
        return stoploss

    def custom_info(self):
        """
        Log custom metrics for performance analysis, such as trade duration or profit distribution.
        """
        return {"example_metric": "value"}

