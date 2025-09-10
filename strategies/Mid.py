from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)


class Mid(IStrategy):
    minimal_roi = {
        "0": 1
    }
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 120
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
    stoploss = -0.25

    is_optimize_32 = True
    buy_bb_period = IntParameter(5, 60, default=15, space='buy', optimize=True)
    buy_supertrend_period = IntParameter(5, 60, default=15, space='buy', optimize=True)
    buy_supertrend_multiplier = DecimalParameter(0.5, 3, default=1, decimals=1, space='buy', optimize=True)
    buy_bb_width_value = DecimalParameter(0.02, 0.2, default=0.02, decimals=2, space='buy', optimize=True)
    buy_add_lost_pct = DecimalParameter(0.01, 0.1, default=0.05, decimals=2, space='buy', optimize=True)

    tpsl_atr_period = IntParameter(5, 60, default=15, space='buy', optimize=True)
    atr_sl_rate = DecimalParameter(0.3, 3, default=0.3, decimals=1, space='buy', optimize=True)
    tpsl_rate = DecimalParameter(0.3, 2.6, default=0.3, decimals=2, space='buy', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bb_bands = pta.bbands(dataframe['close'], length=self.buy_bb_period.value)
        dataframe['bb_lowerband'] = bb_bands.iloc[:, 0]
        dataframe['bb_upperband'] = bb_bands.iloc[:, 2]
        dataframe['bb_middleband'] = bb_bands.iloc[:, 1]
        dataframe['bb_width'] = bb_bands.iloc[:, 3]

        superT = pta.supertrend(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=self.buy_supertrend_period.value, multiplier=self.buy_supertrend_multiplier.value)
        dataframe['supertrend'] = superT.iloc[:, 1]

        atr = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.tpsl_atr_period.value)
        dataframe['atr'] = atr

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
            (dataframe['close'] < dataframe['bb_middleband']) &
            (dataframe['supertrend'] == 1) &
            (dataframe['bb_width'] > self.buy_bb_width_value.value)
        )
        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()






















        if current_profit < 0:
            time_diff = current_time - trade.open_date_utc
            delay_bar = int(time_diff.total_seconds() / (60*15))
            open_candle = dataframe.iloc[-delay_bar] if delay_bar != 0 else current_candle
            if current_candle["close"] < (open_candle["close"]  - open_candle['atr'] * self.atr_sl_rate.value):
                return "lost_sell"
        
        if current_profit > 0:
            time_diff = current_time - trade.open_date_utc
            delay_bar = int(time_diff.total_seconds() / (60*15))
            open_candle = dataframe.iloc[-delay_bar] if delay_bar != 0 else current_candle
            if current_candle["close"] > (open_candle["close"] + open_candle['atr'] * self.tpsl_rate.value):
                return "profit_sell"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe