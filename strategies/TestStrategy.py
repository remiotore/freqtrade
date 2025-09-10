

from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TestStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """


    INTERFACE_VERSION = 2

    buy_params = {
        'adx-enabled': False,
        'adx-value': 24,
        'fastd-enabled': True,
        'fastd-value': 29,
        'mfi-enabled': False,
        'mfi-value': 21,
        'rsi-enabled': False,
        'rsi-value': 29,
        'trigger': 'macd_cross_signal'
    }

    sell_params = {
        'sell-adx-enabled': True,
        'sell-adx-value': 66,
        'sell-fastd-enabled': True,
        'sell-fastd-value': 67,
        'sell-mfi-enabled': True,
        'sell-mfi-value': 82,
        'sell-rsi-enabled': True,
        'sell-rsi-value': 72,
        'sell-trigger': 'sell-sar_reversal'
    }

    minimal_roi = {
        "0": 0.18111,
        "31": 0.02514,
        "85": 0.01181,
        "136": 0
    }

    stoploss = -0.27321

    trailing_stop = False




    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

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
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {

            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

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



        dataframe['adx'] = ta.ADX(dataframe)






























        dataframe['rsi'] = ta.RSI(dataframe)










        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']







        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['mfi'] = ta.MFI(dataframe)





        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )





























        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)



        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']




















































        """

        if self.dp:
            if self.dp.runmode in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe


    @staticmethod
    def buy_strategy_generator(params, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Buy strategy Hyperopt will build and use.
        """
        conditions = []

        if params.get('mfi-enabled'):
            conditions.append(dataframe['mfi'] < params['mfi-value'])
        if params.get('fastd-enabled'):
            conditions.append(dataframe['fastd'] < params['fastd-value'])
        if params.get('adx-enabled'):
            conditions.append(dataframe['adx'] > params['adx-value'])
        if params.get('rsi-enabled'):
            conditions.append(dataframe['rsi'] < params['rsi-value'])

        if 'trigger' in params:
            if params['trigger'] == 'bb_lower':
                conditions.append(dataframe['close'] < dataframe['bb_lowerband'])
            if params['trigger'] == 'macd_cross_signal':
                conditions.append(qtpylib.crossed_above(
                    dataframe['macd'], dataframe['macdsignal']
                ))
            if params['trigger'] == 'sar_reversal':
                conditions.append(qtpylib.crossed_above(
                    dataframe['close'], dataframe['sar']
                ))

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe   


    @staticmethod
    def sell_strategy_generator(params, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Sell strategy Hyperopt will build and use.
        """
        conditions = []

        if params.get('sell-mfi-enabled'):
            conditions.append(dataframe['mfi'] > params['sell-mfi-value'])
        if params.get('sell-fastd-enabled'):
            conditions.append(dataframe['fastd'] > params['sell-fastd-value'])
        if params.get('sell-adx-enabled'):
            conditions.append(dataframe['adx'] < params['sell-adx-value'])
        if params.get('sell-rsi-enabled'):
            conditions.append(dataframe['rsi'] > params['sell-rsi-value'])

        if 'sell-trigger' in params:
            if params['sell-trigger'] == 'sell-bb_upper':
                conditions.append(dataframe['close'] > dataframe['bb_upperband'])
            if params['sell-trigger'] == 'sell-macd_cross_signal':
                conditions.append(qtpylib.crossed_above(
                    dataframe['macdsignal'], dataframe['macd']
                ))
            if params['sell-trigger'] == 'sell-sar_reversal':
                conditions.append(qtpylib.crossed_above(
                    dataframe['sar'], dataframe['close']
                ))

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.buy_strategy_generator(self.buy_params, dataframe, metadata)

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.sell_strategy_generator(self.sell_params, dataframe, metadata)