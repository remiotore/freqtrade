
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.hyper import (CategoricalParameter, DecimalParameter, IntParameter,
                                      RealParameter)







class marketciphertest(IStrategy):


    debuggable_weighted_signal_dataframe = False





    buy_params = {
        'buy_downtrend_oslevel': -37,
        'buy_downtrend_rsi_div_value': 79,
        'buy_downtrend_rsi_divergence_weight': 50,
        'buy_downtrend_total_signal_needed': 70,
        'buy_downtrend_wavetrend_weight': 96,
        'buy_sideways_oslevel': -28,
        'buy_sideways_rsi_div_value': 45,
        'buy_sideways_rsi_divergence_weight': 35,
        'buy_sideways_total_signal_needed': 80,
        'buy_sideways_wavetrend_weight': 5,
        'buy_uptrend_oslevel': -43,
        'buy_uptrend_rsi_div_value': 93,
        'buy_uptrend_rsi_divergence_weight': 84,
        'buy_uptrend_total_signal_needed': 25,
        'buy_uptrend_wavetrend_weight': 91
    }

    sell_params = {
        'sell_downtrend_oblevel': 75.67481,
        'sell_downtrend_rsi_divergence_weight': 90,
        'sell_downtrend_total_signal_needed': 51,
        'sell_downtrend_wavetrend_weight': 30,
        'sell_sideways_oblevel': 1.93351,
        'sell_sideways_rsi_divergence_weight': 59,
        'sell_sideways_total_signal_needed': 36,
        'sell_sideways_wavetrend_weight': 83,
        'sell_uptrend_oblevel': 37.98512,
        'sell_uptrend_rsi_divergence_weight': 99,
        'sell_uptrend_total_signal_needed': 93,
        'sell_uptrend_wavetrend_weight': 43
    }

    minimal_roi = {
        "0": 0.23733,
        "238": 0.20624,
        "846": 0.08939,
        "1834": 0
    }

    stoploss = -0.12447

    trailing_stop = True
    trailing_stop_positive = 0.01658
    trailing_stop_positive_offset = 0.10224
    trailing_only_offset_is_reached = True




    timeframe = '30m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 400



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




    buy___trades_when_downwards = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy___trades_when_sideways = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy___trades_when_upwards = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    buy_uptrend_total_signal_needed = IntParameter(0, 100, default=65, space='buy', optimize=True, load=True)

    buy_uptrend_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_uptrend_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_uptrend_rsi_div_value = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_uptrend_oslevel = \
        IntParameter(-100, 0, default=0, space='buy', optimize=True, load=True)


    buy_sideways_total_signal_needed = IntParameter(0, 100, default=65, space='buy', optimize=True, load=True)

    buy_sideways_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_sideways_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_sideways_rsi_div_value = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_sideways_oslevel = \
        IntParameter(-100, 0, default=0, space='buy', optimize=True, load=True)


    buy_downtrend_total_signal_needed = IntParameter(0, 100, default=65, space='buy', optimize=True, load=True)

    buy_downtrend_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_downtrend_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_downtrend_rsi_div_value = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)

    buy_downtrend_oslevel = \
        IntParameter(-100, 0, default=0, space='buy', optimize=True, load=True)

    sell___trades_when_downwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell___trades_when_sideways = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell___trades_when_upwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)



    sell_uptrend_total_signal_needed = IntParameter(0, 100, default=65, space='sell', optimize=True, load=True)

    sell_uptrend_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    sell_uptrend_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    sell_uptrend_oblevel = \
        RealParameter(0, 100, default=0, space='sell', optimize=True, load=True)



    sell_sideways_total_signal_needed = IntParameter(0, 100, default=65, space='sell', optimize=True, load=True)

    sell_sideways_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    sell_sideways_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    sell_sideways_oblevel = \
        RealParameter(0, 100, default=0, space='sell', optimize=True, load=True)


    sell_downtrend_total_signal_needed = IntParameter(0, 100, default=65, space='sell', optimize=True, load=True)

    sell_downtrend_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    sell_downtrend_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    sell_downtrend_oblevel = \
        RealParameter(0, 100, default=0, space='sell', optimize=True, load=True)









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



        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)  # 14 timeperiods is usually used for ADX

        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=25)

        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=25)

        dataframe['rsi'] = ta.RSI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']  # MACD - Blue TradingView Line (Bullish if on top)
        dataframe['macdsignal'] = macd['macdsignal']  # Signal - Orange TradingView Line (Bearish if on top)

        dataframe['total_buy_signal_strength'] = dataframe['total_sell_signal_strength'] = 0

        dataframe = self.divergences(dataframe)

        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] < dataframe['minus_di']), 'trend'] = 'downwards'
        dataframe.loc[dataframe['adx'] < 20, 'trend'] = 'sideways'
        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] > dataframe['minus_di']), 'trend'] = 'upwards'

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe = self.market_cipher(dataframe)

        dataframe.loc[(dataframe['trend'] == 'downwards'),
                        'wtOversold'] =  (dataframe['wt2'] <= self.buy_downtrend_oslevel.value)
        dataframe.loc[(dataframe['trend'] == 'sideways'),
                        'wtOversold'] = (dataframe['wt2'] <= self.buy_sideways_oslevel.value)
        dataframe.loc[(dataframe['trend'] == 'upwards'),
                        'wtOversold'] = (dataframe['wt2'] <= self.buy_uptrend_oslevel.value)

        dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['wtCrossUp'] & dataframe['wtOversold']),
                        'buy_wavetrend_weight'] = self.buy_downtrend_wavetrend_weight.value
        dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['wtCrossUp'] & dataframe['wtOversold']),
                        'buy_wavetrend_weight'] = self.buy_sideways_wavetrend_weight.value
        dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['wtCrossUp'] & dataframe['wtOversold']),
                        'buy_wavetrend_weight'] = self.buy_uptrend_wavetrend_weight.value
                        
        dataframe['total_buy_signal_strength'] += dataframe['buy_wavetrend_weight']

        dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['bullish_div'] & (dataframe['rsi'] <= self.buy_downtrend_rsi_div_value.value)),
                        'buy_rsi_divergence_weight'] = self.buy_downtrend_rsi_divergence_weight.value
        dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['bullish_div'] & (dataframe['rsi'] <= self.buy_sideways_rsi_div_value.value)),
                        'buy_rsi_divergence_weight'] = self.buy_sideways_rsi_divergence_weight.value
        dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['bullish_div'] & (dataframe['rsi'] <= self.buy_uptrend_rsi_div_value.value)),
                        'buy_rsi_divergence_weight'] = self.buy_uptrend_rsi_divergence_weight.value
                        
        dataframe['total_buy_signal_strength'] += dataframe['buy_rsi_divergence_weight']


        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_buy_signal_strength'] >= self.buy_downtrend_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_buy_signal_strength'] >= self.buy_sideways_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_buy_signal_strength'] >= self.buy_uptrend_total_signal_needed.value)
            ), 'buy'] = 1

        if not self.buy___trades_when_downwards.value:
            dataframe.loc[dataframe['trend'] == 'downwards', 'buy'] = 0
        if not self.buy___trades_when_sideways.value:
            dataframe.loc[dataframe['trend'] == 'sideways', 'buy'] = 0
        if not self.buy___trades_when_upwards.value:
            dataframe.loc[dataframe['trend'] == 'upwards', 'buy'] = 0

        return dataframe

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.market_cipher(dataframe)

        dataframe.loc[(dataframe['trend'] == 'downwards'),
                        'wtOverbought'] =  (dataframe['wt2'] >= self.sell_downtrend_oblevel.value)
        dataframe.loc[(dataframe['trend'] == 'sideways'),
                        'wtOverbought'] = (dataframe['wt2'] >= self.sell_sideways_oblevel.value)
        dataframe.loc[(dataframe['trend'] == 'upwards'),
                        'wtOverbought'] = (dataframe['wt2'] >= self.sell_uptrend_oblevel.value)

        dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['wtCrossDown'] & dataframe['wtOverbought']),
                        'sell_wavetrend_weight'] = self.sell_downtrend_wavetrend_weight.value
        dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['wtCrossDown'] & dataframe['wtOverbought']),
                        'sell_wavetrend_weight'] = self.sell_sideways_wavetrend_weight.value
        dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['wtCrossDown'] & dataframe['wtOverbought']),
                        'sell_wavetrend_weight'] = self.sell_uptrend_wavetrend_weight.value
                        
        dataframe['total_sell_signal_strength'] += dataframe['sell_wavetrend_weight']

        dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['bearish_div']),
                        'sell_rsi_divergence_weight'] = self.sell_downtrend_rsi_divergence_weight.value
        dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['bearish_div']),
                        'sell_rsi_divergence_weight'] = self.sell_sideways_rsi_divergence_weight.value
        dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['bearish_div']),
                        'sell_rsi_divergence_weight'] = self.sell_uptrend_rsi_divergence_weight.value
                        
        dataframe['total_sell_signal_strength'] += dataframe['sell_rsi_divergence_weight']


        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_sell_signal_strength'] >= self.sell_downtrend_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_sell_signal_strength'] >= self.sell_sideways_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_sell_signal_strength'] >= self.sell_uptrend_total_signal_needed.value)
            ), 'sell'] = 1


        return dataframe

    def market_cipher(self, dataframe) -> DataFrame:
            self.n1 = 10 #WT Channel Length
            self.n2 = 21 #WT Average Length

            dataframe['ap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
            dataframe['esa'] = ta.EMA(dataframe['ap'], self.n1)
            dataframe['d'] = ta.EMA((dataframe['ap']-dataframe['esa']).abs(), self.n1)
            dataframe['ci'] = ( dataframe['ap']-dataframe['esa'] ) / (0.015 * dataframe['d'])
            dataframe['tci'] = ta.EMA(dataframe['ci'], self.n2)

            dataframe['wt1'] = dataframe['tci']
            dataframe['wt2'] = ta.SMA(dataframe['wt1'],4)
            
            dataframe['wtVwap'] = dataframe['wt1'] - dataframe['wt2']
            

            dataframe['wtCrossUp'] = dataframe['wt2'] - dataframe['wt1'] <= 0
            dataframe['wtCrossDown'] = dataframe['wt2'] - dataframe['wt1'] >= 0
            dataframe['crossed_above'] = qtpylib.crossed_above(dataframe['wt2'], dataframe['wt1'])
            dataframe['crossed_below'] = qtpylib.crossed_below(dataframe['wt2'], dataframe['wt1'])
            
            return dataframe

    def divergences(self,dataframe) -> DataFrame:
            dataframe['bullish_div'] = (
                                        ( dataframe['close'].shift(4) > dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(3) > dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(2) < dataframe['close'].shift(1) ) & 
                                        ( dataframe['close'].shift(2) < dataframe['close'] )
                                    ) 

            dataframe['bearish_div'] = (
                                        ( dataframe['close'].shift(4) < dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(3) < dataframe['close'].shift(2) ) & 
                                        ( dataframe['close'].shift(2) > dataframe['close'].shift(1) ) & 
                                        ( dataframe['close'].shift(2) > dataframe['close'] )
                                        )

            return dataframe