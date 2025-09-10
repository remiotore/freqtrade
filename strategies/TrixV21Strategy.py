



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (merge_informative_pair, 
                                BooleanParameter, 
                                CategoricalParameter, 
                                DecimalParameter,
                                IStrategy, 
                                IntParameter, 
                                informative)


import talib.abstract as ta
from functools import reduce
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import PairLocks, Trade
from datetime import datetime

class TrixV21Strategy(IStrategy):
    """
    Sources : 
    Cripto Robot : https://www.youtube.com/watch?v=uE04UROWkjs&list=PLpJ7cz_wOtsrqEQpveLc2xKLjOBgy4NfA&index=4
    Github : https://github.com/CryptoRobotFr/TrueStrategy/blob/main/TrixStrategy/Trix_Complete_backtest.ipynb
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
      "0": 0.553,
      "423": 0.144,
      "751": 0.059,
      "1342": 0.02
    }


    stoploss = -0.31

    trailing_stop = False




    buy_params = {
      "buy_stoch_rsi_enabled": True,
      "buy_ema_multiplier": 0.85,
      "buy_ema_src": "open",
      "buy_ema_timeperiod": 10,
      "buy_ema_timeperiod_enabled": True,
      "buy_stoch_rsi": 0.901,
      "buy_trix_signal_timeperiod": 19,
      "buy_trix_signal_type": "trigger",
      "buy_trix_src": "low",
      "buy_trix_timeperiod": 8,
      "buy_ema_guard_multiplier": 0.994,
      "buy_ema_guard_timeperiod": 238
    }

    sell_params = {
      "sell_stoch_rsi": 0.183,
      "sell_stoch_rsi_enabled": True,
      "sell_trix_signal_timeperiod": 19,
      "sell_trix_signal_type": "trailing",
      "sell_trix_src": "high",
      "sell_trix_timeperiod": 10,
      "sell_atr_enabled": True,
      "sell_atr_multiplier": 4.99,
      "sell_atr_timeperiod": 30
    }


    buy_stoch_rsi_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    buy_stoch_rsi = DecimalParameter(0.6, 0.99, decimals=3, default=0.987, space="buy", optimize=False, load=True)

    buy_trix_timeperiod = IntParameter(7, 13, default=12, space="buy", optimize=False, load=True)
    buy_trix_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="buy", optimize=False, load=True)
    buy_trix_signal_timeperiod = IntParameter(19, 25, default=22, space="buy", optimize=False, load=True)
    buy_trix_signal_type = CategoricalParameter(['trailing', 'trigger'], default='trigger', space="buy", optimize=False, load=True)

    buy_ema_timeperiod_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    buy_ema_timeperiod = IntParameter(9, 100, default=21, space="buy", optimize=False, load=True)
    buy_ema_multiplier = DecimalParameter(0.8, 1.2, decimals=2, default=1.00, space="buy", optimize=False, load=True)
    buy_ema_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="buy", optimize=False, load=True)

    buy_ema_guard_timeperiod = IntParameter(150, 250, default=200, space="buy", optimize=True, load=True)
    buy_ema_guard_multiplier = DecimalParameter(0.8, 1.0, decimals=3, default=0.97, space="buy", optimize=True, load=True)

    sell_stoch_rsi_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    sell_stoch_rsi = DecimalParameter(0.01, 0.4, decimals=3, default=0.048, space="sell", optimize=False, load=True)

    sell_trix_timeperiod = IntParameter(7, 11, default=9, space="sell", optimize=False, load=True)
    sell_trix_src = CategoricalParameter(['open', 'high', 'low', 'close'], default='close', space="sell", optimize=False, load=True)
    sell_trix_signal_timeperiod = IntParameter(17, 23, default=19, space="sell", optimize=False, load=True)
    sell_trix_signal_type = CategoricalParameter(['trailing', 'trigger'], default='trailing', space="sell", optimize=False, load=True)

    sell_atr_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    sell_atr_timeperiod = IntParameter(9, 30, default=14, space="sell", optimize=False, load=True)
    sell_atr_multiplier = DecimalParameter(0.7, 9.0, decimals=3, default=4.0, space="sell", optimize=False, load=True)

    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    use_custom_stoploss = True

    startup_candle_count: int = 238

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'trix_b_8': {'color': 'blue'},
            'trix_s_10': {'color': 'orange'},
            'ema_b_signal': {'color': 'red'},
            'stoploss_price': {},
            'ema_guard_238': {},
        },
         'subplots': {
            "TRIX BUY": {
                'trix_b_pct': {'color': 'blue'},
                'trix_b_signal_19': {'color': 'orange'},
            },
            "TRIX SELL": {
                'trix_s_pct': {'color': 'blue'},
                'trix_s_signal_19': {'color': 'orange'},
            },
            "STOCH RSI": {
                'stoch_rsi': {'color': 'blue'},
            },
        }
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if self.sell_atr_enabled.value == False:
            return 1

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        stoploss_price = last_candle['low'] - last_candle[f'atr_{self.sell_atr_timeperiod.value}'] * self.sell_atr_multiplier.value

        if stoploss_price < current_rate:
            return (stoploss_price / current_rate) - 1

        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        if self.buy_stoch_rsi_enabled.value or self.sell_stoch_rsi_enabled.value:

            rsi = ta.RSI(dataframe, timeperiod=14)

            period = 14
            stochrsi  = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
            dataframe['stoch_rsi'] = stochrsi

        for val in self.buy_ema_timeperiod.range:
            dataframe[f'ema_b_{val}'] = ta.EMA(dataframe[self.buy_ema_src.value], timeperiod=val)
        dataframe['ema_b_signal'] = dataframe[f'ema_b_{self.buy_ema_timeperiod.value}'] * self.buy_ema_multiplier.value
        
        for val in self.buy_ema_guard_timeperiod.range:
            dataframe[f'ema_guard_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.buy_trix_timeperiod.range:
            dataframe[f'trix_b_{val}'] = ta.EMA(ta.EMA(ta.EMA(dataframe[self.buy_trix_src.value], timeperiod=val), timeperiod=val), timeperiod=val)
        dataframe['trix_b_pct'] = dataframe[f'trix_b_{self.buy_trix_timeperiod.value}'].pct_change() * 100
        for val in self.buy_trix_signal_timeperiod.range:
            dataframe[f'trix_b_signal_{val}'] = ta.SMA(dataframe['trix_b_pct'], timeperiod=val)

        for val in self.sell_trix_timeperiod.range:
            dataframe[f'trix_s_{val}'] = ta.EMA(ta.EMA(ta.EMA(dataframe[self.sell_trix_src.value], timeperiod=val), timeperiod=val), timeperiod=val)
        dataframe['trix_s_pct'] = dataframe[f'trix_s_{self.sell_trix_timeperiod.value}'].pct_change() * 100
        for val in self.sell_trix_signal_timeperiod.range:
            dataframe[f'trix_s_signal_{val}'] = ta.SMA(dataframe['trix_s_pct'], timeperiod=val)

        for val in self.sell_atr_timeperiod.range:
            dataframe[f'atr_{val}'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=val)
        dataframe['stoploss_price'] = dataframe['low'] - dataframe[f'atr_{self.sell_atr_timeperiod.value}'] * self.sell_atr_multiplier.value

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        conditions = []

        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['close'] > (dataframe[f'ema_guard_{self.buy_ema_guard_timeperiod.value}'] * self.buy_ema_guard_multiplier.value))
        conditions.append(dataframe['trix_s_pct'] > dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}'])
        if self.buy_stoch_rsi_enabled.value:
            conditions.append(dataframe['stoch_rsi'] < self.buy_stoch_rsi.value)
        if self.buy_ema_timeperiod_enabled.value:
            conditions.append(dataframe['close'] > dataframe['ema_b_signal'])
        if self.buy_trix_signal_type.value == 'trailing':
            conditions.append(dataframe['trix_b_pct'] > dataframe[f'trix_b_signal_{self.buy_trix_signal_timeperiod.value}'])

        if self.buy_trix_signal_type.value == 'trigger':
            conditions.append(qtpylib.crossed_above(dataframe['trix_b_pct'], dataframe[f'trix_b_signal_{self.buy_trix_signal_timeperiod.value}']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        conditions = []

        conditions.append(dataframe['volume'] > 0)
        if self.sell_stoch_rsi_enabled.value:
            conditions.append(dataframe['stoch_rsi'] > self.sell_stoch_rsi.value)
        if self.sell_trix_signal_type.value == 'trailing':
            conditions.append(dataframe['trix_s_pct'] < dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}'])

        if self.sell_trix_signal_type.value == 'trigger':
            conditions.append(qtpylib.crossed_below(dataframe['trix_s_pct'], dataframe[f'trix_s_signal_{self.sell_trix_signal_timeperiod.value}']))

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1
        return dataframe