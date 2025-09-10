from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce

class E0V1EFt(IStrategy):
    INTERFACE_VERSION = 3
    minimal_roi = {
        "0": 10
    }

    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 120
    can_short = True

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': True,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    stoploss = -0.18

    is_optimize_32 = True

    long_rsi_fast_32 = IntParameter(20, 70, default=45, space='buy', optimize=is_optimize_32)
    long_rsi_32 = IntParameter(15, 50, default=35, space='buy', optimize=is_optimize_32)
    long_sma15_32 = DecimalParameter(0.900, 1, default=0.96, decimals=2, space='buy', optimize=is_optimize_32)
    long_cti_32 = DecimalParameter(-1, 0, default=-0.58, decimals=2, space='buy', optimize=is_optimize_32)

    short_rsi_fast_32 = IntParameter(30, 80, default=55, space='buy', optimize=is_optimize_32)
    short_rsi_32 = IntParameter(50, 85, default=65, space='buy', optimize=is_optimize_32)
    short_sma15_32 = DecimalParameter(1, 1.1, default=1.04, decimals=2, space='buy', optimize=is_optimize_32)
    short_cti_32 = DecimalParameter(0, 1, default=0.58, decimals=2, space='buy', optimize=is_optimize_32)

    exit_long_fastx = IntParameter(50, 100, default=75, space='sell', optimize=True)
    exit_short_fastx = IntParameter(0, 50, default=25, space='sell', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions_long = []
        conditions_short = []
        dataframe.loc[:, 'enter_tag'] = ''

        long_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.long_rsi_fast_32.value) &
                (dataframe['rsi'] > self.long_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.long_sma15_32.value) &
                (dataframe['cti'] < self.long_cti_32.value)
        )

        conditions_long.append(long_1)
        dataframe.loc[long_1, 'enter_tag'] += 'long_1'

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_long),
                'enter_long'] = 1
        
        short_1 = (
                (dataframe['rsi_slow'] > dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] > self.short_rsi_fast_32.value) &
                (dataframe['rsi'] < self.short_rsi_32.value) &
                (dataframe['close'] > dataframe['sma_15'] * self.short_sma15_32.value) &
                (dataframe['cti'] > self.short_cti_32.value)
        )

        conditions_short.append(short_1)
        dataframe.loc[short_1, 'enter_tag'] += 'short_1'

        if conditions_short:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions_short),
                'enter_short'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        current_candle = dataframe.iloc[-1].squeeze()

        if current_time - timedelta(minutes=90) < trade.open_date_utc:
            if current_profit >= 0.08:
                return "fastk_profit_exit_fast"

        if current_time - timedelta(hours=3) > trade.open_date_utc:
            if trade.is_short:
                if (current_candle["fastk"] <= 30) and (current_profit >= 0):
                    return "fastk_profit_exit_short_delay"
            else:
                if (current_candle["fastk"] >= 70) and (current_profit >= 0):
                    return "fastk_profit_exit_long_delay"

        if current_time - timedelta(hours=4) > trade.open_date_utc:
            if current_profit > -0.06:
                return "fastk_loss_exit_delay"

        if current_profit > 0:
            if current_candle["fastk"] > self.exit_long_fastx.value and (not trade.is_short):
                return "fastk_profit_exit_long"
            if current_candle["fastk"] < self.exit_short_fastx.value and trade.is_short:
                return "fastk_profit_exit_short"











        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')
        dataframe.loc[(), ['exit_short', 'exit_tag']] = (0, 'short_out')

        return dataframe
