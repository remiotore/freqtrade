

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair


from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class StandardStrategy(IStrategy):


    INTERFACE_VERSION = 2

    MARKET_CAP_REFERENCE_PAIR = 'BTC/USDT'

    custom_info = {}


    minimal_roi = {
        '0': 100
    }


    stoploss = -0.30

    use_custom_stoploss = True

    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 20


    protections = []

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'ema7': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            'MACD': {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            'RSI': {
                'rsi': {'color': 'red'},
            }
        }
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        is_bull_market = True

        if self.custom_info and self.MARKET_CAP_REFERENCE_PAIR in self.custom_info and trade:


            if self.dp and self.dp.runmode.value in ('backtest', 'hyperopt'):
                sar = self.custom_info[self.MARKET_CAP_REFERENCE_PAIR]['sar_1w'].loc[current_time]['sar_1w']
                tema = self.custom_info[self.MARKET_CAP_REFERENCE_PAIR]['tema_1w'].loc[current_time]['tema_1w']

            else:

                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=self.MARKET_CAP_REFERENCE_PAIR, timeframe=self.timeframe)




                sar = dataframe['sar_1w'].iat[-1]
                tema = dataframe['tema_1w'].iat[-1]

            if tema is not None and sar is not None:
                is_bull_market = tema > sar

        if current_profit > 5.00:
            return -0.2
        elif current_profit > 4.00:
            return stoploss_from_open(3.50, current_profit)
        elif current_profit > 3.50:
            return stoploss_from_open(3.00, current_profit)
        elif current_profit > 3.00:
            return stoploss_from_open(2.50, current_profit)
        elif current_profit > 2.50:
            return stoploss_from_open(2.00, current_profit)
        elif current_profit > 2.00:
            return stoploss_from_open(1.50, current_profit)
        elif current_profit > 1.50:
            return stoploss_from_open(1.25, current_profit)
        elif current_profit > 1.25:
            return stoploss_from_open(1.00, current_profit)
        elif current_profit > 1.00:
            return stoploss_from_open(0.75, current_profit)
        elif current_profit > 0.75:
            return stoploss_from_open(0.50, current_profit)
        elif current_profit > 0.50 and is_bull_market:
            return stoploss_from_open(0.25, current_profit)
        elif current_profit > 0.25 and is_bull_market:
            return stoploss_from_open(0.05, current_profit)
        elif -0.05 < current_profit < 0.05 and not is_bull_market:
            if current_time - timedelta(hours=24*7) > trade.open_date_utc:
                return -0.0125
            elif current_time - timedelta(hours=24*5) > trade.open_date_utc:
                return -0.025
            elif current_time - timedelta(hours=24*3) > trade.open_date_utc:
                return -0.05

        return -1

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()

        informative_pairs = [(pair, '1w') for pair in pairs]

        informative_pairs += [(self.MARKET_CAP_REFERENCE_PAIR, '1w')]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:

            return dataframe

        inf_tf = '1w'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        informative['sar'] = ta.SAR(dataframe)

        informative['tema'] = ta.TEMA(dataframe, timeperiod=9)





        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)

        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['bb_width'] = (
                (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        )

        dataframe['bb_width_past_1'] = (
                (dataframe['bb_upperband'].shift(1) - dataframe['bb_lowerband'].shift(1)) / dataframe['bb_middleband'].shift(1)
        )

        dataframe['bb_width_past_2'] = (
                (dataframe['bb_upperband'].shift(2) - dataframe['bb_lowerband'].shift(2)) / dataframe['bb_middleband'].shift(2)
        )

        dataframe['bb_width_past_3'] = (
                (dataframe['bb_upperband'].shift(3) - dataframe['bb_lowerband'].shift(3)) / dataframe['bb_middleband'].shift(3)
        )

        dataframe['bb_width_past_4'] = (
                (dataframe['bb_upperband'].shift(4) - dataframe['bb_lowerband'].shift(4)) / dataframe['bb_middleband'].shift(4)
        )

        dataframe['bb_width_past_5'] = (
                (dataframe['bb_upperband'].shift(5) - dataframe['bb_lowerband'].shift(5)) / dataframe['bb_middleband'].shift(5)
        )

        if not metadata['pair'] in self.custom_info:

            self.custom_info[metadata['pair']] = {}

        if self.dp.runmode.value in ('backtest', 'hyperopt'):

            self.custom_info[metadata['pair']]['sar_1w'] = dataframe[['date', 'sar_1w']].set_index('date')
            self.custom_info[metadata['pair']]['tema_1w'] = dataframe[['date', 'tema_1w']].set_index('date')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['tema'] > dataframe['sar']) &
                    (dataframe['rsi'] > 70) &
                    (dataframe['rsi'] < 90) &
                    (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                    ((dataframe['bb_width_past_1'] / dataframe['bb_width_past_2']) > 0.975) &
                    ((dataframe['bb_width_past_2'] / dataframe['bb_width_past_1']) < 1.0257) &
                    ((dataframe['bb_width_past_2'] / dataframe['bb_width_past_3']) > 0.975) &
                    ((dataframe['bb_width_past_3'] / dataframe['bb_width_past_2']) < 1.0257) &
                    ((dataframe['bb_width_past_3'] / dataframe['bb_width_past_4']) > 0.975) &
                    ((dataframe['bb_width_past_4'] / dataframe['bb_width_past_3']) < 1.0257) &
                    ((dataframe['bb_width_past_4'] / dataframe['bb_width_past_5']) > 0.975) &
                    ((dataframe['bb_width_past_5'] / dataframe['bb_width_past_4']) < 1.0257) &
                    (dataframe['bb_width'] / dataframe['bb_width_past_1'] > 1.14) &
                    (dataframe['bb_width'] / dataframe['bb_width_past_1'] < 2.00) &
                    (dataframe['volume'] > 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['tema'] < dataframe['sar']) &
                    (qtpylib.crossed_above(dataframe['ema7'], dataframe['tema'])) &
                    (dataframe['bb_width_past_1'] / dataframe['bb_width'] > 1.20) &
                    (dataframe['volume'] > 0)
            ),
            'sell'] = 1

        return dataframe
