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

class EVA1_Optimized(IStrategy):
    minimal_roi = { "0": 1 }
    timeframe = '1h'
    process_only_new_candles = True
    startup_candle_count = 120
    stoploss = -0.15  # Reduced stop-loss to minimize risk

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
    }

    # Optimizable Parameters
    buy_rsi = IntParameter(20, 50, default=35, space='buy', optimize=True)
    buy_sma_ratio = DecimalParameter(0.90, 1, default=0.95, decimals=2, space='buy', optimize=True)
    buy_cti = DecimalParameter(-1, 0, default=-0.6, decimals=2, space='buy', optimize=True)
    buy_ema_confirmation = IntParameter(10, 50, default=20, space='buy', optimize=True)
    sell_rsi = IntParameter(50, 80, default=70, space='sell', optimize=True)
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='sell', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['close'] < dataframe['sma_15'] * self.buy_sma_ratio.value) &
            (dataframe['cti'] < self.buy_cti.value) &
            (dataframe['close'] > dataframe['ema_50']) &
            (dataframe['ema_50'] > dataframe['ema_200'])  # Trend confirmation
        )
        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] = 'buy_1'

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1]

        if current_profit >= 0.07:
            return "take_profit"  # Take profit at 7%

        if current_candle['rsi'] > self.sell_rsi.value:
            return "rsi_sell"

        atr_stop_loss = trade.open_rate - (dataframe['atr'].iloc[-1] * self.atr_multiplier.value)
        if current_rate < atr_stop_loss:
            return "atr_stop_loss"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe
