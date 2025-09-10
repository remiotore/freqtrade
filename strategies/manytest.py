
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, IntParameter






class manytest(IStrategy):
    """























    """


    debuggable_weighted_signal_dataframe = False

    sell_unclogger_enabled = False





    buy_params = {
        'buy__downwards_trend_total_signal_needed': 43,
        'buy__sideways_trend_total_signal_needed': 82,
        'buy__upwards_trend_total_signal_needed': 17,
        'buy_downwards_trend_adx_strong_up_weight': 97,
        'buy_downwards_trend_bollinger_bands_weight': 13,
        'buy_downwards_trend_ema_long_golden_cross_weight': 50,
        'buy_downwards_trend_ema_short_golden_cross_weight': 26,
        'buy_downwards_trend_macd_weight': 52,
        'buy_downwards_trend_rsi_weight': 67,
        'buy_downwards_trend_sma_long_golden_cross_weight': 77,
        'buy_downwards_trend_sma_short_golden_cross_weight': 38,
        'buy_downwards_trend_vwap_cross_weight': 13,
        'buy_rsi_divergence_weight': 90,
        'buy_sideways_trend_adx_strong_up_weight': 63,
        'buy_sideways_trend_bollinger_bands_weight': 71,
        'buy_sideways_trend_ema_long_golden_cross_weight': 36,
        'buy_sideways_trend_ema_short_golden_cross_weight': 47,
        'buy_sideways_trend_macd_weight': 76,
        'buy_sideways_trend_rsi_weight': 90,
        'buy_sideways_trend_sma_long_golden_cross_weight': 7,
        'buy_sideways_trend_sma_short_golden_cross_weight': 66,
        'buy_sideways_trend_vwap_cross_weight': 43,
        'buy_upwards_trend_adx_strong_up_weight': 17,
        'buy_upwards_trend_bollinger_bands_weight': 21,
        'buy_upwards_trend_ema_long_golden_cross_weight': 44,
        'buy_upwards_trend_ema_short_golden_cross_weight': 80,
        'buy_upwards_trend_macd_weight': 33,
        'buy_upwards_trend_rsi_weight': 43,
        'buy_upwards_trend_sma_long_golden_cross_weight': 69,
        'buy_upwards_trend_sma_short_golden_cross_weight': 60,
        'buy_upwards_trend_vwap_cross_weight': 1,
        'buy_wavetrend_weight': 61
    }

    sell_params = {
        'sell___trades_when_downwards': False,
        'sell___trades_when_sideways': False,
        'sell___trades_when_upwards': False,
        'sell__downwards_trend_total_signal_needed': 7,
        'sell__sideways_trend_total_signal_needed': 40,
        'sell__upwards_trend_total_signal_needed': 80,
        'sell_downwards_trend_adx_strong_down_weight': 23,
        'sell_downwards_trend_bollinger_bands_weight': 69,
        'sell_downwards_trend_ema_long_death_cross_weight': 69,
        'sell_downwards_trend_ema_short_death_cross_weight': 21,
        'sell_downwards_trend_macd_weight': 7,
        'sell_downwards_trend_rsi_weight': 72,
        'sell_downwards_trend_sma_long_death_cross_weight': 69,
        'sell_downwards_trend_sma_short_death_cross_weight': 43,
        'sell_downwards_trend_vwap_cross_weight': 50,
        'sell_rsi_divergence_weight': 7,
        'sell_sideways_trend_adx_strong_down_weight': 42,
        'sell_sideways_trend_bollinger_bands_weight': 54,
        'sell_sideways_trend_ema_long_death_cross_weight': 30,
        'sell_sideways_trend_ema_short_death_cross_weight': 11,
        'sell_sideways_trend_macd_weight': 79,
        'sell_sideways_trend_rsi_weight': 21,
        'sell_sideways_trend_sma_long_death_cross_weight': 43,
        'sell_sideways_trend_sma_short_death_cross_weight': 57,
        'sell_sideways_trend_vwap_cross_weight': 86,
        'sell_upwards_trend_adx_strong_down_weight': 86,
        'sell_upwards_trend_bollinger_bands_weight': 87,
        'sell_upwards_trend_ema_long_death_cross_weight': 31,
        'sell_upwards_trend_ema_short_death_cross_weight': 64,
        'sell_upwards_trend_macd_weight': 75,
        'sell_upwards_trend_rsi_weight': 35,
        'sell_upwards_trend_sma_long_death_cross_weight': 2,
        'sell_upwards_trend_sma_short_death_cross_weight': 41,
        'sell_upwards_trend_vwap_cross_weight': 78,
        'sell_wavetrend_weight': 44
    }

    minimal_roi = {
        "0": 0.64208,
        "371": 0.18428,
        "791": 0.08118,
        "1818": 0
    }

    stoploss = -0.05811

    trailing_stop = True
    trailing_stop_positive = 0.01062
    trailing_stop_positive_offset = 0.03918
    trailing_only_offset_is_reached = False




    timeframe = '1h'

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

    plot_config = {
        'main_plot': {

            'sma9': {'color': '#2c05f6'},
            'sma50': {'color': '#19038a'},
            'sma200': {'color': '#0d043b'},
            'ema9': {'color': '#12e5a6'},
            'ema50': {'color': '#0a8963'},
            'ema200': {'color': '#074b36'},
            'bb_upperband': {'color': '#6f1a7b'},
            'bb_lowerband': {'color': '#6f1a7b'},
            'vwap': {'color': '#727272'}
        },
        'subplots': {

            'MACD (Moving Average Convergence Divergence)': {
                'macd': {'color': '#19038a'},
                'macdsignal': {'color': '#ae231c'}
            },
            'ADX (Average Directional Index) + Plus & Minus Directions': {
                'adx': {'color': '#6f1a7b'},
                'plus_di': {'color': '#0ad628'},
                'minus_di': {'color': '#ae231c'}
            },
            'RSI (Relative Strength Index)': {
                'rsi': {'color': '#7fba3c'}
            }
        }
    }


























    buy___trades_when_downwards = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy___trades_when_sideways = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy___trades_when_upwards = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)




    buy__downwards_trend_total_signal_needed = IntParameter(0, 100, default=65, space='buy', optimize=True, load=True)

    buy_downwards_trend_adx_strong_up_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_rsi_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_macd_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_sma_short_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_ema_short_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_sma_long_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_ema_long_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_bollinger_bands_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_vwap_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)



    buy__sideways_trend_total_signal_needed = IntParameter(0, 100, default=65, space='buy', optimize=True, load=True)

    buy_sideways_trend_adx_strong_up_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_rsi_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_macd_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_sma_short_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_ema_short_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_sma_long_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_ema_long_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_bollinger_bands_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_vwap_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)



    buy__upwards_trend_total_signal_needed = IntParameter(0, 200, default=65, space='buy', optimize=True, load=True)

    buy_upwards_trend_adx_strong_up_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_rsi_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_macd_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_sma_short_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_ema_short_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_sma_long_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_ema_long_golden_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_bollinger_bands_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_vwap_cross_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)











    sell___trades_when_downwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell___trades_when_sideways = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)
    sell___trades_when_upwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=True, load=True)



    sell__downwards_trend_total_signal_needed = IntParameter(0, 100, default=65, space='sell', optimize=True, load=True)

    sell_downwards_trend_adx_strong_down_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_rsi_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_macd_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_sma_short_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_ema_short_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_sma_long_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_ema_long_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_bollinger_bands_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_vwap_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)



    sell__sideways_trend_total_signal_needed = IntParameter(0, 100, default=65, space='sell', optimize=True, load=True)

    sell_sideways_trend_adx_strong_down_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_rsi_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_macd_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_sma_short_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_ema_short_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_sma_long_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_ema_long_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_bollinger_bands_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_vwap_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)



    sell__upwards_trend_total_signal_needed = IntParameter(0, 100, default=65, space='sell', optimize=True, load=True)

    sell_upwards_trend_adx_strong_down_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_rsi_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_macd_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_sma_short_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_ema_short_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_sma_long_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_ema_long_death_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_bollinger_bands_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_vwap_cross_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    buy_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    sell_wavetrend_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)

    buy_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='buy', optimize=True, load=True)
    sell_rsi_divergence_weight = \
        IntParameter(0, 100, default=0, space='sell', optimize=True, load=True)









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




        dataframe['sma9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)


        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)  # timeperiod is expressed in candles
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']



        dataframe['vwap'] = qtpylib.vwap(dataframe)



        if self.debuggable_weighted_signal_dataframe:
            dataframe['adx_strong_up_weighted_buy_signal'] = dataframe['adx_strong_down_weighted_sell_signal'] = 0
            dataframe['rsi_weighted_buy_signal'] = dataframe['rsi_weighted_sell_signal'] = 0
            dataframe['macd_weighted_buy_signal'] = dataframe['macd_weighted_sell_signal'] = 0
            dataframe['sma_short_golden_cross_weighted_buy_signal'] = 0
            dataframe['sma_short_death_cross_weighted_sell_signal'] = 0
            dataframe['ema_short_golden_cross_weighted_buy_signal'] = 0
            dataframe['ema_short_death_cross_weighted_sell_signal'] = 0
            dataframe['sma_long_golden_cross_weighted_buy_signal'] = 0
            dataframe['sma_long_death_cross_weighted_sell_signal'] = 0
            dataframe['ema_long_golden_cross_weighted_buy_signal'] = 0
            dataframe['ema_long_death_cross_weighted_sell_signal'] = 0
            dataframe['bollinger_bands_weighted_buy_signal'] = dataframe['bollinger_bands_weighted_sell_signal'] = 0
            dataframe['vwap_cross_weighted_buy_signal'] = dataframe['vwap_cross_weighted_sell_signal'] = 0

        dataframe['total_buy_signal_strength'] = dataframe['total_sell_signal_strength'] = 0

        self.n1 = 10 #WT Channel Length
        self.n2 = 21 #WT Average Length
        dataframe = self.market_cipher(dataframe)

        dataframe = self.divergences(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] < dataframe['minus_di']), 'trend'] = 'downwards'
        dataframe.loc[dataframe['adx'] < 20, 'trend'] = 'sideways'
        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] > dataframe['minus_di']), 'trend'] = 'upwards'


        if self.debuggable_weighted_signal_dataframe:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = self.buy_downwards_trend_adx_strong_up_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = self.buy_sideways_trend_adx_strong_up_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = self.buy_upwards_trend_adx_strong_up_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['adx_strong_up_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_downwards_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_sideways_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_upwards_trend_rsi_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['rsi_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_downwards_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_sideways_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_upwards_trend_macd_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['macd_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_sma_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_sma_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_sma_short_golden_cross_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['sma_short_golden_cross_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_ema_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_ema_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_ema_short_golden_cross_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['ema_short_golden_cross_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_sma_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_sma_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_sma_long_golden_cross_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['sma_long_golden_cross_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_ema_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_ema_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_ema_long_golden_cross_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['ema_long_golden_cross_weighted_buy_signal']


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_downwards_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_sideways_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_upwards_trend_bollinger_bands_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['bollinger_bands_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = self.buy_downwards_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = self.buy_sideways_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = self.buy_upwards_trend_vwap_cross_weight.value
            dataframe['total_buy_signal_strength'] += dataframe['vwap_cross_weighted_buy_signal']

        else:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += self.buy_downwards_trend_adx_strong_up_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += self.buy_sideways_trend_adx_strong_up_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += self.buy_upwards_trend_adx_strong_up_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_downwards_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_sideways_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_upwards_trend_rsi_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_downwards_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_sideways_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_upwards_trend_macd_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += self.buy_downwards_trend_sma_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += self.buy_sideways_trend_sma_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += self.buy_upwards_trend_sma_short_golden_cross_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += self.buy_downwards_trend_ema_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += self.buy_sideways_trend_ema_short_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += self.buy_upwards_trend_ema_short_golden_cross_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += self.buy_downwards_trend_sma_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += self.buy_sideways_trend_sma_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += self.buy_upwards_trend_sma_long_golden_cross_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += self.buy_downwards_trend_ema_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += self.buy_sideways_trend_ema_long_golden_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += self.buy_upwards_trend_ema_long_golden_cross_weight.value


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += self.buy_downwards_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += self.buy_sideways_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += self.buy_upwards_trend_bollinger_bands_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += self.buy_downwards_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += self.buy_sideways_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += self.buy_upwards_trend_vwap_cross_weight.value

            dataframe.loc[(dataframe['wtCrossUp'] & dataframe['wtOversold']), 'total_buy_signal_strength'] += self.buy_wavetrend_weight.value 

            dataframe.loc[(dataframe['bullish_div']), 'total_buy_signal_strength'] += self.buy_rsi_divergence_weight.value 

        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_buy_signal_strength'] >= self.buy__downwards_trend_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_buy_signal_strength'] >= self.buy__sideways_trend_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_buy_signal_strength'] >= self.buy__upwards_trend_total_signal_needed.value)
            ), 'buy'] = 1

        if not self.buy___trades_when_downwards.value:
            dataframe.loc[dataframe['trend'] == 'downwards', 'buy'] = 0
        if not self.buy___trades_when_sideways.value:
            dataframe.loc[dataframe['trend'] == 'sideways', 'buy'] = 0
        if not self.buy___trades_when_upwards.value:
            dataframe.loc[dataframe['trend'] == 'upwards', 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] < dataframe['minus_di']), 'trend'] = 'downwards'
        dataframe.loc[dataframe['adx'] < 20, 'trend'] = 'sideways'
        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] > dataframe['minus_di']), 'trend'] = 'upwards'


        if self.debuggable_weighted_signal_dataframe:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_downwards_trend_adx_strong_down_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_sideways_trend_adx_strong_down_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_upwards_trend_adx_strong_down_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['adx_strong_down_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_downwards_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_sideways_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_upwards_trend_rsi_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['rsi_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_downwards_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_sideways_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_upwards_trend_macd_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['macd_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_sma_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_sma_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_sma_short_death_cross_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['sma_short_death_cross_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_ema_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_ema_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_ema_short_death_cross_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['ema_short_death_cross_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_sma_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_sma_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_sma_long_death_cross_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['sma_long_death_cross_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_ema_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_ema_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_ema_long_death_cross_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['ema_long_death_cross_weighted_sell_signal']


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_downwards_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_sideways_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_upwards_trend_bollinger_bands_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['bollinger_bands_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = self.sell_downwards_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = self.sell_sideways_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = self.sell_upwards_trend_vwap_cross_weight.value
            dataframe['total_sell_signal_strength'] += dataframe['vwap_cross_weighted_sell_signal']

        else:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += self.sell_downwards_trend_adx_strong_down_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += self.sell_sideways_trend_adx_strong_down_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += self.sell_upwards_trend_adx_strong_down_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_downwards_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_sideways_trend_rsi_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_upwards_trend_rsi_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_downwards_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_sideways_trend_macd_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_upwards_trend_macd_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += self.sell_downwards_trend_sma_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += self.sell_sideways_trend_sma_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += self.sell_upwards_trend_sma_short_death_cross_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += self.sell_downwards_trend_ema_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += self.sell_sideways_trend_ema_short_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += self.sell_upwards_trend_ema_short_death_cross_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += self.sell_downwards_trend_sma_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += self.sell_sideways_trend_sma_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += self.sell_upwards_trend_sma_long_death_cross_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += self.sell_downwards_trend_ema_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += self.sell_sideways_trend_ema_long_death_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += self.sell_upwards_trend_ema_long_death_cross_weight.value


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_bollinger_bands_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_bollinger_bands_weight.value

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += self.sell_downwards_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += self.sell_sideways_trend_vwap_cross_weight.value
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += self.sell_upwards_trend_vwap_cross_weight.value

            dataframe.loc[(dataframe['wtCrossDown'] & dataframe['wtOverbought']), 'total_sell_signal_strength'] += self.sell_wavetrend_weight.value 

            dataframe.loc[(dataframe['bearish_div']), 'total_sell_signal_strength'] += self.sell_rsi_divergence_weight.value 

        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_sell_signal_strength'] >= self.sell__downwards_trend_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_sell_signal_strength'] >= self.sell__sideways_trend_total_signal_needed.value)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_sell_signal_strength'] >= self.sell__upwards_trend_total_signal_needed.value)
            ), 'sell'] = 1

        if not self.sell___trades_when_downwards.value:
            dataframe.loc[dataframe['trend'] == 'downwards', 'sell'] = 0
        if not self.sell___trades_when_sideways.value:
            dataframe.loc[dataframe['trend'] == 'sideways', 'sell'] = 0
        if not self.sell___trades_when_upwards.value:
            dataframe.loc[dataframe['trend'] == 'upwards', 'sell'] = 0

        return dataframe

    def market_cipher(self, dataframe) -> DataFrame:


            osLevel = -60
            obLevel = 30
            dataframe['ap'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
            dataframe['esa'] = ta.EMA(dataframe['ap'], self.n1)
            dataframe['d'] = ta.EMA((dataframe['ap']-dataframe['esa']).abs(), self.n1)
            dataframe['ci'] = ( dataframe['ap']-dataframe['esa'] ) / (0.015 * dataframe['d'])
            dataframe['tci'] = ta.EMA(dataframe['ci'], self.n2)

            dataframe['wt1'] = dataframe['tci']
            dataframe['wt2'] = ta.SMA(dataframe['wt1'],4)
            
            dataframe['wtVwap'] = dataframe['wt1'] - dataframe['wt2']
            dataframe['wtOversold'] =   dataframe['wt2'] <= osLevel
            dataframe['wtOverbought'] =   dataframe['wt2'] >= obLevel
            

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