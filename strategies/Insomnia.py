





from typing import Dict
from pandas import DataFrame
from datetime import datetime
from typing import Optional
from freqtrade.strategy import (DecimalParameter,
                                IntParameter, IStrategy, informative,  stoploss_from_open)
from freqtrade.persistence import Trade
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal


import talib.abstract as ta

    
class Insomnia(IStrategy):
    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 10,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 12,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
        ]


























    INTERFACE_VERSION = 3

    timeframe = '30m'

    can_short: bool = False

    levarage_input = 4.0

    minimal_roi = {
        "0": 0.4,
        "150": 0.2, 
        "300": 0.0
    }


    stoploss = -0.034

    trailing_stop = False
    trailing_stop_positive = 0.219
    trailing_stop_positive_offset = 0.284
    trailing_only_offset_is_reached = False

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 250


    buy_params = {
        "buy_rsi": 14,
        "buy_rsi_1": 14,
        "buy_rsi_4": 14,
        "buy_rsi_12": 14,
        "buy_sma_1": 200,
        "buy_sma_4": 200,
        "buy_sma_12": 200,
        "buy_rsi_compare_long": 32,
        "buy_rsi_compare_short": 68,
        "buy_compare_vol_long": 3,
        "buy_compare_vol_short": 3,
        "buy_compare_rsi_12_long": 49,
        "buy_compare_rsi_12_short": 49,
        "buy_compare_rsi_1_long": 35,
        "buy_compare_rsi_1_short": 65,
    }

    sell_params = {
        "sell_rsi_compare_long": 65,
        "sell_rsi_compare_short": 30,
    }

    buy_rsi = IntParameter(8, 20, default=buy_params['buy_rsi'], space="buy")

    buy_rsi_1 = IntParameter(8, 20, default=buy_params['buy_rsi_1'], space="buy", optimize=False)
    buy_rsi_4 = IntParameter(8, 20, default=buy_params['buy_rsi_4'], space="buy", optimize=False)
    buy_rsi_12 = IntParameter(8, 20, default=buy_params['buy_rsi_12'], space="buy", optimize=False)

    buy_sma_1 = IntParameter(150, 250, default=buy_params['buy_sma_1'], space="buy")
    buy_sma_4 = IntParameter(150, 250, default=buy_params['buy_sma_4'], space="buy")
    buy_sma_12 = IntParameter(150, 250, default=buy_params['buy_sma_12'], space="buy")


    buy_rsi_compare_long = IntParameter(20, 49, default=buy_params['buy_rsi_compare_long'], space="buy", optimize=False)
    buy_rsi_compare_short = IntParameter(51, 80, default=buy_params['buy_rsi_compare_short'], space="buy")

    buy_compare_vol_long = DecimalParameter(0.1, 5, decimals=1, default=buy_params['buy_compare_vol_long'], space="buy", optimize=False)
    buy_compare_vol_short = DecimalParameter(0.1, 5, decimals=1, default=buy_params['buy_compare_vol_short'], space="buy", optimize=False)

    buy_compare_rsi_12_long = IntParameter(20, 49, default=buy_params['buy_compare_rsi_12_long'], space="buy", optimize=False)
    buy_compare_rsi_12_short = IntParameter(49, 80, default=buy_params['buy_compare_rsi_12_short'], space="buy", optimize=False)

    buy_compare_rsi_1_long = IntParameter(20, 49, default=buy_params['buy_compare_rsi_1_long'], space="buy")
    buy_compare_rsi_1_short = IntParameter(51, 80, default=buy_params['buy_compare_rsi_1_short'], space="buy")

    sell_rsi_compare_long = IntParameter(65, 70, default=buy_params['buy_compare_rsi_1_short'], space="sell")
    sell_rsi_compare_short = IntParameter(30, 50, default=buy_params['buy_compare_rsi_1_short'], space="sell")

    use_custom_stoploss = True

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    @property
    def plot_config(self):
        return {

            'main_plot': {
                f'sma_{self.buy_sma_1.value}_1h': {'color': 'green'},
                f'sma_{self.buy_sma_4.value}_4h': {'color': 'brown'},
                f'sma_{self.buy_sma_12.value}_12h': {'color': 'red'},
            },
            'subplots': {

                "RSI": {
                    f'rsi_{self.buy_rsi.value}': {'color': 'green'},
                    f'rsi_{self.buy_rsi_1.value}_1h': {'color': 'brown'},
                    f'rsi_{self.buy_rsi_12.value}_12h': {'color': 'red'},
                },
                "Volume": {
                    'volume_usdt': {'color': 'green', 'type': 'bar'},
                }
            }
        }

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.buy_rsi_1.range:
            dataframe[f'rsi_{val}'] = ta.RSI(dataframe, timeperiod=val)
        for val in self.buy_sma_1.range:
            dataframe[f'sma_{val}'] = ta.SMA(dataframe, timeperiod=val)
        return dataframe

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.buy_rsi_4.range:
            dataframe[f'rsi_{val}'] = ta.RSI(dataframe, timeperiod=val)
        for val in self.buy_sma_4.range:
            dataframe[f'sma_{val}'] = ta.SMA(dataframe, timeperiod=val)
        return dataframe

    @informative('12h')
    def populate_indicators_12h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.buy_rsi_12.range:
            dataframe[f'rsi_{val}'] = ta.RSI(dataframe, timeperiod=val)
        for val in self.buy_sma_12.range:
            dataframe[f'sma_{val}'] = ta.SMA(dataframe, timeperiod=val)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for val in self.buy_rsi.range:
            dataframe[f'rsi_{val}'] = ta.RSI(dataframe, timeperiod=val)
        dataframe['volume_cum'] = dataframe['volume'].rolling(window=48).sum()
        dataframe['volume_usdt'] = dataframe['volume_cum'] * dataframe['close']


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            ((dataframe[f'rsi_{self.buy_rsi.value}'] < self.buy_rsi_compare_long.value) &
             (dataframe[f'volume_usdt'] > self.buy_compare_vol_long.value * 1000000) &
             (dataframe[f'rsi_{self.buy_rsi_12.value}_12h'] > self.buy_compare_rsi_12_long.value) &
             (dataframe[f'rsi_{self.buy_rsi_1.value}_1h'] < self.buy_compare_rsi_1_long.value) &
             (dataframe['close'] > dataframe[f'sma_{self.buy_sma_12.value}_12h']) &
             (dataframe['close'] > dataframe[f'sma_{self.buy_sma_4.value}_4h']) &
             (dataframe['close'] > dataframe[f'sma_{self.buy_sma_1.value}_1h'])
            ),
            'enter_long'] = 1


        dataframe.loc[
            ((dataframe[f'rsi_{self.buy_rsi.value}'] > self.buy_rsi_compare_short.value) &
             (dataframe[f'volume_usdt'] > self.buy_compare_vol_short.value * 1000000) &
             (dataframe[f'rsi_{self.buy_rsi_12.value}_12h'] < self.buy_compare_rsi_12_short.value) &
             (dataframe[f'rsi_{self.buy_rsi_1.value}_1h'] > self.buy_compare_rsi_1_short.value) &
             (dataframe['close'] < dataframe[f'sma_{self.buy_sma_12.value}_12h']) &
             (dataframe['close'] < dataframe[f'sma_{self.buy_sma_4.value}_4h']) &
             (dataframe['close'] < dataframe[f'sma_{self.buy_sma_1.value}_1h'])
            ),
            'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe[f'rsi_{self.buy_rsi.value}'] > self.sell_rsi_compare_long.value),
            'exit_long'] = 1

        dataframe.loc[
            (dataframe[f'rsi_{self.buy_rsi.value}'] < self.sell_rsi_compare_short.value),
            'exit_short'] = 1


        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        if self.levarage_input > max_leverage:
            return max_leverage

        return self.levarage_input

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> Optional[float]:

        if current_profit > 0.12:
            return stoploss_from_open(0.02, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.15:
            return stoploss_from_open(0.05, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.25:
            return stoploss_from_open(0.18, current_profit, is_short=trade.is_short, leverage=trade.leverage)
        elif current_profit > 0.3:
            return stoploss_from_open(0.25, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        return None