# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series, DatetimeIndex, merge
# --------------------------------
import talib.abstract as ta
import pandas_ta as pta
import numpy as np
import pandas as pd  # noqa
import warnings, datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import stoploss_from_open, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from functools import reduce
from datetime import timedelta

pd.options.mode.chained_assignment = None  # default='warn'

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, ou %R, est un oscillateur d'analyse technique indiquant la position du cours de clôture
       par rapport aux plus hauts et aux plus bas sur une période donnée.
       L'oscillateur est exprimé sur une échelle négative de -100 (le plus bas) à 0 (le plus haut).
    """
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return WR * -100

class BestSpot(IStrategy):
    INTERFACE_VERSION = 2

    # Paramètres optimisables
    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)
    r_threshold = DecimalParameter(-80, -40, default=-61.3, space='buy', optimize=True)
    cti_threshold = DecimalParameter(-1, 0, default=-0.715, space='buy', optimize=True)

    @property
    def protections(self):
        return [
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 60,
                "trade_limit": 1,
                "stop_duration_candles": 60,
                "required_profit": -0.05
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            }
        ]

    minimal_roi = {
        "0": 1
    }
    cc = {}

    # Stoploss dynamique via ATR
    stoploss = -0.25  # Valeur par défaut remplacée par custom_stoploss

    # Trailing stop activé
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 999

    plot_config = {
        'main_plot': {
            "mama": {'color': '#d0da3e'},
            "fama": {'color': '#da3eb8'},
            "kama": {'color': '#3edad8'},
            "ema_50": {'color': '#1f77b4'},
            "ema_200": {'color': '#ff7f0e'}
        },
        "subplots": {
            "fastk": {
                "fastk": {'color': '#da3e3e'}
            },
            "ATR": {
                "atr": {'color': '#2ca02c'}
            },
            "Williams %R": {
                "r_14": {'color': '#9467bd'}
            }
        }
    }

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                          current_rate: float, current_profit: float, **kwargs) -> float:
        # Si le profit est supérieur ou égal à 4 %, appliquer un stoploss très serré
        if current_profit >= 0.04:
            return -0.002
        # Si la position est ouverte depuis plus de 1,5 jours, forcer la sortie avec un stoploss très serré

        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Volatilité
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # PCT CHANGE
        dataframe['change'] = 100 / dataframe['open'] * dataframe['close'] - 100

        # MAMA, FAMA, KAMA
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.25, 0.025)
        dataframe['mama_diff'] = ((dataframe['mama'] - dataframe['fama']) / dataframe['hl2'])
        dataframe['kama'] = ta.KAMA(dataframe['close'], 84)

        # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # Moyennes mobiles
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_300'] = ta.EMA(dataframe, timeperiod=300)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        buy_condition = (
            (dataframe['kama'] > dataframe['fama']) &
            (dataframe['fama'] > dataframe['mama'] * 0.981) &
            (dataframe['r_14'] < self.r_threshold.value) &  # Paramètre optimisable
            (dataframe['mama_diff'] < -0.025) &
            (dataframe['cti'] < self.cti_threshold.value) &  # Paramètre optimisable
            (dataframe['close'].rolling(48).max() >= dataframe['close'] * 1.05) &
            (dataframe['close'].rolling(288).max() >= dataframe['close'] * 1.125) &
            (dataframe['rsi_84'] < 60) &
            (dataframe['rsi_112'] < 60) &
            (dataframe['ema_200'] > dataframe['ema_300'])  # Filtre tendance activé
        )

        conditions.append(buy_condition)
        dataframe.loc[buy_condition, 'enter_tag'] += 'trend_confirmed'

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # Gestion du temps (timezone-aware)
        trade_open_date = trade.open_date.replace(tzinfo=current_time.tzinfo) if trade.open_date.tzinfo is None else trade.open_date
        trade_age = current_time - trade_open_date

        # Sortie forcée après 60h
        if trade_age >= timedelta(hours=60):
            return "60h_exit"

        # Mise à jour du suivi de prix
        state = self.cc
        pc = state.get(trade.id, {'date': current_candle['date'], 'open': current_candle['close'], 'high': current_candle['close'], 'low': current_candle['close'], 'close': current_rate, 'volume': 0})
        if current_candle['date'] != pc['date']:
            pc['date'] = current_candle['date']
            pc['high'] = current_candle['close']
            pc['low'] = current_candle['close']
            pc['open'] = current_candle['close']
            pc['close'] = current_rate
        if current_rate > pc['high']:
            pc['high'] = current_rate
        if current_rate < pc['low']:
            pc['low'] = current_rate
        if current_rate != pc['close']:
            pc['close'] = current_rate

        state[trade.id] = pc

        # Sortie sur surchauffe (Stochastic)
        if current_profit > 0:
            # if min_profit <= -0.015:
            if current_time > pc['date'] + timedelta(minutes=9) + timedelta(seconds=55):
                df = dataframe.copy()
                df = df._append(pc, ignore_index = True)
                stoch_fast = ta.STOCHF(df, 5, 3, 0, 3, 0)
                df['fastk'] = stoch_fast['fastk']
                cc = df.iloc[-1].squeeze()
                if cc["fastk"] > self.sell_fastx.value:
                    return "fastk_profit_sell_2"
            else:
                if current_candle["fastk"] > self.sell_fastx.value:
                    return "fastk_profit_sell"


        return None