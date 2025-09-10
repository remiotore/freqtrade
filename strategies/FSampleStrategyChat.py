# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
# --------------------------------

class FSampleStrategyChat(IStrategy):
    timeframe = '5m'
    informative_timeframe = '1h'

    # Futures config
    position_adjustment_enable = True
    can_short = True

    # RSI thresholds
    rsi_overbought = 70
    rsi_oversold = 30

    # BTC pair for market condition filtering
    btc_pair = "BTC/USDT:USDT"

    @informative('1h', btc_pair)
    def populate_indicators_btc(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for BTC 1h data
        Columns will be automatically prefixed as '1h_BTC/USDT:USDT_*'
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['btc_close_ma'] = ta.SMA(dataframe['close'], timeperiod=168)  # 1-week MA on 1h data
        dataframe['rsi_peak'] = (dataframe['rsi'] > self.rsi_overbought)
        dataframe['rsi_dip'] = (dataframe['rsi'] < self.rsi_oversold)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Main timeframe indicators
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        btc_prefix = f"1h_{self.btc_pair}_"

        # Long entry: RSI dip + above BTC trend
        dataframe.loc[
            (dataframe['rsi'] < self.rsi_oversold) &
            (dataframe[f'{btc_prefix}close'] > dataframe[f'{btc_prefix}btc_close_ma']) &
            (dataframe[f'{btc_prefix}rsi_dip']) &
            (dataframe['ema_fast'] > dataframe['ema_slow']),
            'enter_long'
        ] = 1

        # Short entry: RSI peak + below BTC trend
        dataframe.loc[
            (dataframe['rsi'] > self.rsi_overbought) &
            (dataframe[f'{btc_prefix}close'] < dataframe[f'{btc_prefix}btc_close_ma']) &
            (dataframe[f'{btc_prefix}rsi_peak']) &
            (dataframe['ema_fast'] < dataframe['ema_slow']),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Tight exit rules to avoid over-holding
        """
        # Long exit: RSI rebound above 50 or EMA cross down
        dataframe.loc[
            (dataframe['rsi'] > 50) |
            (dataframe['ema_fast'] < dataframe['ema_slow']),
            'exit_long'
        ] = 1

        # Short exit: RSI drop below 50 or EMA cross up
        dataframe.loc[
            (dataframe['rsi'] < 50) |
            (dataframe['ema_fast'] > dataframe['ema_slow']),
            'exit_short'
        ] = 1

        return dataframe
