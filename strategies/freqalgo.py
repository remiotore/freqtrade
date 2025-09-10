from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import DecimalParameter, IntParameter
from pandas_ta.core import ta
import pandas as pd
import numpy as np
from datetime import datetime
from freqtrade.persistence import Trade
from functools import reduce
import talib.abstract as ta_lib

class AlgoXStrategy(IStrategy):
    """
    Freqtrade strategia oparta na ALGOX Pine Script dla timeframe 15m.
    Dla trybu spot: tylko long positions (buy/sell), bez shortów.
    Wejścia na podstawie crossoverów Heikin Ashi lub Renko EMA.
    Wyjścia z TP/SL na podstawie ATR.
    """

    INTERFACE_VERSION = 3

    # Ustawienie timeframe na 15 minut
    timeframe = "15m"

    # Minimal ROI dla spot
    minimal_roi = {"0": 0.01}  # Minimalny ROI, np. 1%

    # Stoploss (domyślny)
    stoploss = -0.05  # -5% SL domyślny, ale używamy custom

    # Trailing stop (opcjonalnie)
    trailing_stop = False

    # Parametry optymalizacyjne
    startup_candle_count = 500  # Max bars back
    tp_type = "ATR"  # 'ATR', 'Trailing', 'Options' - dla spot używamy ATR dla TP/SL
    setup_type = "Open/Close"  # 'Open/Close' lub 'Renko'
    atr_length = IntParameter(10, 30, default=20, space="buy")
    profit_factor = DecimalParameter(1.0, 4.0, default=2.5, space="sell")
    stop_factor = DecimalParameter(0.5, 2.0, default=1.0, space="sell")
    ema1_length = IntParameter(1, 5, default=2, space="buy")
    ema2_length = IntParameter(5, 15, default=10, space="buy")
    atr_len_renko = IntParameter(1, 10, default=3, space="buy")

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Heikin Ashi candles
        heikinashi = ta.ha(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
        dataframe['ha_open'] = heikinashi['HA_open']
        dataframe['ha_close'] = heikinashi['HA_close']

        # ATR
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_length.value)

        if self.setup_type == "Renko":
            # Symulacja Renko (prosta, bo Freqtrade nie ma wbudowanego Renko; używamy ATR-based bricks)
            brick_size = dataframe['atr'].rolling(window=self.atr_len_renko.value).mean()
            # Prosta symulacja Renko close/open
            dataframe['renko_close'] = dataframe['close']  # Placeholder; dostosuj jeśli masz lepszą implementację
            dataframe['renko_open'] = dataframe['open']
            dataframe['ema1'] = ta.ema(dataframe['renko_close'], length=self.ema1_length.value)
            dataframe['ema2'] = ta.ema(dataframe['renko_close'], length=self.ema2_length.value)
        else:
            # Dla Open/Close używamy Heikin Ashi
            dataframe['ema1'] = ta.ema(dataframe['ha_close'], length=self.ema1_length.value)
            dataframe['ema2'] = ta.ema(dataframe['ha_close'], length=self.ema2_length.value)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions_long = []

        if self.setup_type == "Open/Close":
            # Crossover ha_close > ha_open
            conditions_long.append(dataframe['ha_close'] > dataframe['ha_open'])
            conditions_long.append(dataframe['ha_close'].shift(1) <= dataframe['ha_open'].shift(1))
        elif self.setup_type == "Renko":
            # Crossover ema1 > ema2
            conditions_long.append(dataframe['ema1'] > dataframe['ema2'])
            conditions_long.append(dataframe['ema1'].shift(1) <= dataframe['ema2'].shift(1))

        dataframe.loc[reduce(lambda x, y: x & y, conditions_long), 'enter_long'] = 1
        dataframe['enter_tag'] = 'long_entry'

        # Brak shortów dla spot
        dataframe['enter_short'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions_short = []

        if self.setup_type == "Open/Close":
            conditions_short.append(dataframe['ha_close'] < dataframe['ha_open'])
            conditions_short.append(dataframe['ha_close'].shift(1) >= dataframe['ha_open'].shift(1))
        elif self.setup_type == "Renko":
            conditions_short.append(dataframe['ema1'] < dataframe['ema2'])
            conditions_short.append(dataframe['ema1'].shift(1) >= dataframe['ema2'].shift(1))

        dataframe.loc[reduce(lambda x, y: x & y, conditions_short), 'exit_long'] = 1
        dataframe['exit_tag'] = 'long_exit'

        dataframe['exit_short'] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Custom stoploss oparty na ATR SL.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        sl_price = trade.open_rate - (self.stop_factor.value * last_candle['atr'])
        sl_offset = (sl_price / trade.open_rate) - 1

        return sl_offset

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> str:
        """
        Custom exit dla TP1/TP2/TP3 oparty na ATR.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        tp1_price = trade.open_rate + (self.profit_factor.value * last_candle['atr'])
        tp2_price = trade.open_rate + (2 * self.profit_factor.value * last_candle['atr'])
        tp3_price = trade.open_rate + (3 * self.profit_factor.value * last_candle['atr'])

        if current_rate >= tp3_price:
            return 'tp3_exit'
        elif current_rate >= tp2_price:
            return 'tp2_exit'
        elif current_rate >= tp1_price:
            return 'tp1_exit'

        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs):
        """
        Partial exits dla TP1/TP2/TP3.
        """
        exit_reason = self.custom_exit(
            pair=trade.pair, trade=trade, current_time=current_time,
            current_rate=current_rate, current_profit=current_profit
        )
        if exit_reason == 'tp1_exit':
            return - (trade.amount * 0.5)  # 50% qty
        elif exit_reason == 'tp2_exit':
            return - (trade.amount * 0.3)  # 30% qty
        elif exit_reason == 'tp3_exit':
            return - (trade.amount * 0.2)  # 20% qty

        return None
