
import logging
from datetime import datetime, timedelta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np  # noqa
import pandas as pd  # noqa
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, CategoricalParameter, IntParameter, RealParameter
from pandas import DataFrame

logger = logging.getLogger(__name__)






class maninew(IStrategy):
    """























    """


    debuggable_weighted_signal_dataframe = False


    use_mgm_logging = False
    mgm_log_levels_enabled = {
        'info': True,
        'warning': True,
        'error': True,
        'debug': True


    }

    use_custom_stoploss = True





    buy_params = {
        'buy__downwards_trend_total_signal_needed': 32,
        'buy__sideways_trend_total_signal_needed': 96,
        'buy__upwards_trend_total_signal_needed': 27,
        'buy_downwards_trend_adx_strong_up_weight': 65,
        'buy_downwards_trend_bollinger_bands_weight': 93,
        'buy_downwards_trend_ema_long_golden_cross_weight': 98,
        'buy_downwards_trend_ema_short_golden_cross_weight': 29,
        'buy_downwards_trend_macd_weight': 96,
        'buy_downwards_trend_rsi_weight': 41,
        'buy_downwards_trend_sma_long_golden_cross_weight': 64,
        'buy_downwards_trend_sma_short_golden_cross_weight': 35,
        'buy_downwards_trend_vwap_cross_weight': 64,
        'buy_sideways_trend_adx_strong_up_weight': 25,
        'buy_sideways_trend_bollinger_bands_weight': 86,
        'buy_sideways_trend_ema_long_golden_cross_weight': 50,
        'buy_sideways_trend_ema_short_golden_cross_weight': 63,
        'buy_sideways_trend_macd_weight': 43,
        'buy_sideways_trend_rsi_weight': 85,
        'buy_sideways_trend_sma_long_golden_cross_weight': 73,
        'buy_sideways_trend_sma_short_golden_cross_weight': 85,
        'buy_sideways_trend_vwap_cross_weight': 9,
        'buy_upwards_trend_adx_strong_up_weight': 86,
        'buy_upwards_trend_bollinger_bands_weight': 18,
        'buy_upwards_trend_ema_long_golden_cross_weight': 61,
        'buy_upwards_trend_ema_short_golden_cross_weight': 94,
        'buy_upwards_trend_macd_weight': 69,
        'buy_upwards_trend_rsi_weight': 4,
        'buy_upwards_trend_sma_long_golden_cross_weight': 18,
        'buy_upwards_trend_sma_short_golden_cross_weight': 86,
        'buy_upwards_trend_vwap_cross_weight': 12
    }

    sell_params = {
        'sell___unclogger_minimal_losing_trade_duration_minutes': 53,
        'sell___unclogger_minimal_losing_trades_open': 2,
        'sell___unclogger_percentage_open_trades_losing': 16,
        'sell___unclogger_trend_lookback_candles_window': 0,
        'sell__downwards_trend_total_signal_needed': 77,
        'sell__sideways_trend_total_signal_needed': 68,
        'sell__upwards_trend_total_signal_needed': 85,
        'sell_downwards_trend_adx_strong_down_weight': 31,
        'sell_downwards_trend_bollinger_bands_weight': 47,
        'sell_downwards_trend_ema_long_death_cross_weight': 4,
        'sell_downwards_trend_ema_short_death_cross_weight': 45,
        'sell_downwards_trend_macd_weight': 48,
        'sell_downwards_trend_rsi_weight': 78,
        'sell_downwards_trend_sma_long_death_cross_weight': 23,
        'sell_downwards_trend_sma_short_death_cross_weight': 86,
        'sell_downwards_trend_vwap_cross_weight': 79,
        'sell_sideways_trend_adx_strong_down_weight': 48,
        'sell_sideways_trend_bollinger_bands_weight': 7,
        'sell_sideways_trend_ema_long_death_cross_weight': 79,
        'sell_sideways_trend_ema_short_death_cross_weight': 85,
        'sell_sideways_trend_macd_weight': 86,
        'sell_sideways_trend_rsi_weight': 88,
        'sell_sideways_trend_sma_long_death_cross_weight': 48,
        'sell_sideways_trend_sma_short_death_cross_weight': 65,
        'sell_sideways_trend_vwap_cross_weight': 94,
        'sell_upwards_trend_adx_strong_down_weight': 73,
        'sell_upwards_trend_bollinger_bands_weight': 84,
        'sell_upwards_trend_ema_long_death_cross_weight': 85,
        'sell_upwards_trend_ema_short_death_cross_weight': 32,
        'sell_upwards_trend_macd_weight': 61,
        'sell_upwards_trend_rsi_weight': 4,
        'sell_upwards_trend_sma_long_death_cross_weight': 73,
        'sell_upwards_trend_sma_short_death_cross_weight': 1,
        'sell_upwards_trend_vwap_cross_weight': 98
    }

    minimal_roi = {
        "0": 0.26931,
        "243": 0.17413,
        "752": 0.0596,
        "908": 0
    }

    stoploss = -0.21617

    trailing_stop = True
    trailing_stop_positive = 0.25261
    trailing_stop_positive_offset = 0.33494
    trailing_only_offset_is_reached = False




    custom_info = {
        'open_trades': {},
        'trend_indicator': {}
    }

    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 400






    precision = 1

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
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=False)
    buy___trades_when_sideways = \
        CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=False)
    buy___trades_when_upwards = \
        CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=False)



    buy__downwards_trend_total_signal_needed = IntParameter(0, int(100 * precision), default=65, space='buy',
                                                            optimize=True, load=True)

    buy_downwards_trend_adx_strong_up_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_sma_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_ema_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_sma_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_ema_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_downwards_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)



    buy__sideways_trend_total_signal_needed = IntParameter(0, int(100 * precision), default=65, space='buy',
                                                           optimize=True, load=True)

    buy_sideways_trend_adx_strong_up_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_sma_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_ema_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_sma_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_ema_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_sideways_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)



    buy__upwards_trend_total_signal_needed = IntParameter(0, int(100 * precision), default=65, space='buy',
                                                          optimize=True, load=True)

    buy_upwards_trend_adx_strong_up_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_sma_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_ema_short_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_sma_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_ema_long_golden_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)
    buy_upwards_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='buy', optimize=True, load=True)











    sell___trades_when_downwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___trades_when_sideways = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___trades_when_upwards = \
        CategoricalParameter([True, False], default=False, space='sell', optimize=False, load=False)



    sell__downwards_trend_total_signal_needed = IntParameter(0, int(100 * precision), default=65, space='sell',
                                                             optimize=True, load=True)

    sell_downwards_trend_adx_strong_down_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_sma_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_ema_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_sma_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_ema_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_downwards_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)



    sell__sideways_trend_total_signal_needed = IntParameter(0, int(100 * precision), default=65, space='sell',
                                                            optimize=True, load=True)

    sell_sideways_trend_adx_strong_down_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_sma_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_ema_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_sma_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_ema_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_sideways_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)



    sell__upwards_trend_total_signal_needed = IntParameter(0, int(100 * precision), default=65, space='sell',
                                                           optimize=True, load=True)

    sell_upwards_trend_adx_strong_down_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_rsi_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_macd_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_sma_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_ema_short_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_sma_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_ema_long_death_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_bollinger_bands_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell_upwards_trend_vwap_cross_weight = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)




    sell___unclogger_enabled = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___unclogger_enabled_when_downwards = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___unclogger_enabled_when_sideways = \
        CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=False)
    sell___unclogger_enabled_when_upwards = \
        CategoricalParameter([True, False], default=False, space='sell', optimize=False, load=False)
    sell___unclogger_minimal_losing_trades_open = \
        IntParameter(0, int(15 * precision), default=0, space='sell', optimize=True, load=True)
    sell___unclogger_minimal_losing_trade_duration_minutes = \
        IntParameter(0, int(300 * precision), default=0, space='sell', optimize=True, load=True)
    sell___unclogger_percentage_open_trades_losing = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)
    sell___unclogger_trend_lookback_candles_window = \
        IntParameter(0, int(100 * precision), default=0, space='sell', optimize=True, load=True)

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
        you are using. Let uncomment only the indicator you are using in MoniGoMani
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for MoniGoMani
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



        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] < dataframe['minus_di']), 'trend'] = 'downwards'
        dataframe.loc[dataframe['adx'] < 20, 'trend'] = 'sideways'
        dataframe.loc[(dataframe['adx'] > 20) & (dataframe['plus_di'] > dataframe['minus_di']), 'trend'] = 'upwards'




        if self.sell___unclogger_enabled.value and self.dp and (
                self.dp.runmode.value in ('backtest', 'hyperopt')):
            self.mgm_logger('info', 'Custom Information Storage', 'Storing whole \'trend\' indicator for ' +
                            'pair (' + metadata['pair'] + ') in custom_info')

            if metadata['pair'] not in self.custom_info['trend_indicator']:
                self.custom_info['trend_indicator'][metadata['pair']] = {}
            self.custom_info['trend_indicator'][metadata['pair']] = \
                dataframe[['date', 'trend']].dropna().copy().set_index('date')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """


        if self.debuggable_weighted_signal_dataframe:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = \
                self.buy_downwards_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = \
                self.buy_sideways_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'adx_strong_up_weighted_buy_signal'] = \
                self.buy_upwards_trend_adx_strong_up_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['adx_strong_up_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'rsi_weighted_buy_signal'] = self.buy_upwards_trend_rsi_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['rsi_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'macd_weighted_buy_signal'] = self.buy_upwards_trend_macd_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['macd_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['sma_short_golden_cross_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['ema_short_golden_cross_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['sma_long_golden_cross_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_golden_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['ema_long_golden_cross_weighted_buy_signal']


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'bollinger_bands_weighted_buy_signal'] = \
                self.buy_upwards_trend_bollinger_bands_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['bollinger_bands_weighted_buy_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = \
                self.buy_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = \
                self.buy_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_buy_signal'] = \
                self.buy_upwards_trend_vwap_cross_weight.value / self.precision
            dataframe['total_buy_signal_strength'] += dataframe['vwap_cross_weighted_buy_signal']

        else:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += \
                self.buy_downwards_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += \
                self.buy_sideways_trend_adx_strong_up_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'total_buy_signal_strength'] += \
                self.buy_upwards_trend_adx_strong_up_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['rsi'], 30),
                          'total_buy_signal_strength'] += self.buy_upwards_trend_rsi_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] > dataframe['macdsignal']),
                          'total_buy_signal_strength'] += self.buy_upwards_trend_macd_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_sma_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma9'], dataframe[
                'sma50']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_sma_short_golden_cross_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_ema_short_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema9'], dataframe[
                'ema50']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_ema_short_golden_cross_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_sma_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['sma50'], dataframe[
                'sma200']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_sma_long_golden_cross_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_ema_long_golden_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['ema50'], dataframe[
                'ema200']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_ema_long_golden_cross_weight.value / self.precision


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['close'], dataframe[
                'bb_lowerband']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_bollinger_bands_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += \
                self.buy_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += \
                self.buy_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_above(dataframe['vwap'], dataframe[
                'close']), 'total_buy_signal_strength'] += \
                self.buy_upwards_trend_vwap_cross_weight.value / self.precision

        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_buy_signal_strength'] >= self.buy__downwards_trend_total_signal_needed.value /
                     self.precision)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_buy_signal_strength'] >= self.buy__sideways_trend_total_signal_needed.value /
                     self.precision)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_buy_signal_strength'] >= self.buy__upwards_trend_total_signal_needed.value /
                     self.precision)
            ), 'buy'] = 1

        if not self.buy___trades_when_downwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'downwards', 'buy'] = 0
        if not self.buy___trades_when_sideways.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'sideways', 'buy'] = 0
        if not self.buy___trades_when_upwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'upwards', 'buy'] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """


        if self.debuggable_weighted_signal_dataframe:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_downwards_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_sideways_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'adx_strong_down_weighted_sell_signal'] = \
                self.sell_upwards_trend_adx_strong_down_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['adx_strong_down_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'rsi_weighted_sell_signal'] = self.sell_upwards_trend_rsi_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['rsi_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'macd_weighted_sell_signal'] = self.sell_upwards_trend_macd_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['macd_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'sma_short_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_sma_short_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['sma_short_death_cross_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'ema_short_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_ema_short_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['ema_short_death_cross_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'sma_long_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_sma_long_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['sma_long_death_cross_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'ema_long_death_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_ema_long_death_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['ema_long_death_cross_weighted_sell_signal']


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'bollinger_bands_weighted_sell_signal'] = \
                self.sell_upwards_trend_bollinger_bands_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['bollinger_bands_weighted_sell_signal']

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = \
                self.sell_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = \
                self.sell_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'vwap_cross_weighted_sell_signal'] = \
                self.sell_upwards_trend_vwap_cross_weight.value / self.precision
            dataframe['total_sell_signal_strength'] += dataframe['vwap_cross_weighted_sell_signal']

        else:

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += \
                self.sell_downwards_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += \
                self.sell_sideways_trend_adx_strong_down_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['adx'] > 25),
                          'total_sell_signal_strength'] += \
                self.sell_upwards_trend_adx_strong_down_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_downwards_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_sideways_trend_rsi_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['rsi'], 70),
                          'total_sell_signal_strength'] += self.sell_upwards_trend_rsi_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_downwards_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_sideways_trend_macd_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & (dataframe['macd'] < dataframe['macdsignal']),
                          'total_sell_signal_strength'] += self.sell_upwards_trend_macd_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_sma_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma9'], dataframe[
                'sma50']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_sma_short_death_cross_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_ema_short_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema9'], dataframe[
                'ema50']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_ema_short_death_cross_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_sma_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['sma50'], dataframe[
                'sma200']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_sma_long_death_cross_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_ema_long_death_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['ema50'], dataframe[
                'ema200']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_ema_long_death_cross_weight.value / self.precision


            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_bollinger_bands_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['close'], dataframe[
                'bb_upperband']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_bollinger_bands_weight.value / self.precision

            dataframe.loc[(dataframe['trend'] == 'downwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += \
                self.sell_downwards_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'sideways') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += \
                self.sell_sideways_trend_vwap_cross_weight.value / self.precision
            dataframe.loc[(dataframe['trend'] == 'upwards') & qtpylib.crossed_below(dataframe['vwap'], dataframe[
                'close']), 'total_sell_signal_strength'] += \
                self.sell_upwards_trend_vwap_cross_weight.value / self.precision

        dataframe.loc[
            (
                    (dataframe['trend'] == 'downwards') &
                    (dataframe['total_sell_signal_strength'] >=
                     self.sell__downwards_trend_total_signal_needed.value / self.precision)
            ) | (
                    (dataframe['trend'] == 'sideways') &
                    (dataframe['total_sell_signal_strength'] >=
                     self.sell__sideways_trend_total_signal_needed.value / self.precision)
            ) | (
                    (dataframe['trend'] == 'upwards') &
                    (dataframe['total_sell_signal_strength'] >=
                     self.sell__upwards_trend_total_signal_needed.value / self.precision)
            ), 'sell'] = 1

        if not self.sell___trades_when_downwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'downwards', 'sell'] = 0
        if not self.sell___trades_when_sideways.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'sideways', 'sell'] = 0
        if not self.sell___trades_when_upwards.value / self.precision:
            dataframe.loc[dataframe['trend'] == 'upwards', 'sell'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Open Trade Unclogger:
        ---------------------
        Override Sell Signal: When enabled attempts to unclog the bot when it's stuck with losing trades & unable to
        trade more new trades. This custom_stoploss function should be able to work in tandem with Trailing stoploss.
        It will only unclog a losing trade when all of following checks have been full-filled:
        => Check if everything in custom_storage is up to date with all_open_trades
        => Check if there are enough losing trades open for unclogging to occur
        => Check if there is a losing trade open for the pair currently being ran through the MoniGoMani loop
        => Check if trade has been open for X minutes (long enough to give it a recovery chance)
        => Check if total open trades losing % is met
        => Check if open_trade's trend changed negatively during past X candles
        Please configurable/hyperoptable in the sell_params dictionary under the hyperopt results copy/paste section.
        Only used when use_custom_stoploss & sell_params['sell___unclogger_enabled'] are both set to True.
        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break MoniGoMani.
        :return float: New stoploss value, relative to the current-rate
        """
        open_trade_unclogger = 'Open Trade Unclogger'
        custom_information_storage = 'Custom Information Storage'
        garbage_collector = custom_information_storage + ' Garbage Collector'

        if self.sell___unclogger_enabled.value:
            try:



                is_live_or_dry_run = self.dp and self.dp.runmode.value in ('live', 'dry_run')
                if is_live_or_dry_run:
                    self.mgm_logger('debug', custom_information_storage,
                                    'Fetching all currently open trades during Dry/Live Run')

                    all_open_trades = Trade.get_trades([Trade.is_open.is_(True)]).order_by(Trade.open_date).all()

                else:
                    self.mgm_logger('debug', custom_information_storage,
                                    'Fetching all currently open trades during BackTesting/HyperOpting')
                    all_open_trades = trade.trades_open

                self.mgm_logger('debug', custom_information_storage,
                                'Up-to-date open trades (' + str(len(all_open_trades)) + ') fetched!')
                self.mgm_logger('debug', custom_information_storage, 'all_open_trades contents: ' +
                                repr(all_open_trades))

                for open_trade in all_open_trades:
                    if str(open_trade.pair) == str(pair):
                        if str(open_trade.pair) not in self.custom_info['open_trades']:
                            self.custom_info['open_trades'][str(open_trade.pair)] = {}
                        self.custom_info['open_trades'][str(open_trade.pair)]['trade'] = str(open_trade)
                        self.custom_info['open_trades'][str(open_trade.pair)]['current_profit'] = current_profit


                        self.mgm_logger('info', custom_information_storage, 'Storing trade + current profit/loss + ' +
                                        'open date for pair (' + str(pair) + ') in custom_info')
                        break



                if len(all_open_trades) < len(self.custom_info['open_trades']):
                    garbage_trade_amount = len(self.custom_info['open_trades']) - len(all_open_trades)
                    self.mgm_logger('info', garbage_collector, 'Old open trade garbage detected for ' +
                                    str(garbage_trade_amount) + ' trades, starting cleanup')

                    for garbage_trade in range(garbage_trade_amount):
                        for stored_trade in self.custom_info['open_trades']:
                            pair_still_open = False
                            for open_trade in all_open_trades:
                                if str(stored_trade) == str(open_trade.pair):
                                    self.mgm_logger('debug', garbage_collector, 'Open trade found, no action needed ' +
                                                    'for pair (' + stored_trade + ') in custom_info')
                                    pair_still_open = True
                                    break

                            if not pair_still_open:
                                self.mgm_logger('info', garbage_collector, 'No open trade found for pair (' +
                                                stored_trade + '), removing from custom_info')
                                self.custom_info['open_trades'].pop(stored_trade)
                                self.mgm_logger('debug', garbage_collector,
                                                'Successfully removed garbage_trade ' + str(garbage_trade) +
                                                ' from custom_info!')
                                break

















                elif len(all_open_trades) > len(self.custom_info['open_trades']):
                    self.mgm_logger('warning', custom_information_storage, 'Open trades (' +
                                    str(len(self.custom_info['open_trades'])) + ') in custom_storage do not match yet '
                                    + 'with trades in live open trades (' + str(len(all_open_trades)) +
                                    ') aborting unclogger for now!')
                    return self.stoploss

                self.mgm_logger('debug', custom_information_storage, 'Open trades (' + str(len(
                    self.custom_info['open_trades'])) + ') in custom_info updated successfully!')
                self.mgm_logger('debug', custom_information_storage, 'custom_info[\'open_trades\'] contents: ' +
                                repr(self.custom_info['open_trades']))


                self.mgm_logger('debug', open_trade_unclogger,
                                'Running trough all checks to see if unclogging is needed')

                self.mgm_logger('debug', open_trade_unclogger,
                                'Fetching all currently losing_open_trades from custom information storage')
                losing_open_trades = {}
                for stored_trade in self.custom_info['open_trades']:
                    stored_current_profit = self.custom_info['open_trades'][stored_trade]['current_profit']
                    if stored_current_profit < 0:
                        if not str(pair) in losing_open_trades:
                            losing_open_trades[str(stored_trade)] = {}
                        losing_open_trades[str(stored_trade)] = stored_current_profit
                self.mgm_logger('debug', open_trade_unclogger, 'Fetched losing_open_trades (' +
                                str(len(losing_open_trades)) + ') from custom information storage!')

                if len(losing_open_trades) < (self.sell___unclogger_minimal_losing_trades_open.value / self.precision):
                    self.mgm_logger('debug', open_trade_unclogger, 'No unclogging needed! ' +
                                    'Not enough losing trades currently open!')
                else:
                    self.mgm_logger('debug', open_trade_unclogger, 'Enough losing trades detected! '
                                    + 'Proceeding to the next check!')

                    if pair not in losing_open_trades:
                        self.mgm_logger('debug', open_trade_unclogger, 'No unclogging needed! Currently checked pair ('
                                        + pair + ') is not making a loss at this point in time!')
                    else:
                        self.mgm_logger('debug', open_trade_unclogger, 'Currently checked pair (' + pair +
                                        ') is losing! Proceeding to the next check!')

                        if is_live_or_dry_run:
                            current_datetime_to_use = datetime.utcnow()
                        else:
                            current_datetime_to_use = current_time

                        self.mgm_logger('debug', open_trade_unclogger, 'Trade open time : ' + str(
                            trade.open_date_utc.replace(tzinfo=None)))
                        self.mgm_logger('debug', open_trade_unclogger, 'Minimal open time: ' + str(
                            current_datetime_to_use.replace(tzinfo=None) -
                            timedelta(minutes=(self.sell___unclogger_minimal_losing_trade_duration_minutes.value /
                                               self.precision))))

                        if trade.open_date_utc.replace(tzinfo=None) > (
                                current_datetime_to_use.replace(tzinfo=None) -
                                timedelta(minutes=(self.sell___unclogger_minimal_losing_trade_duration_minutes.value /
                                                   self.precision))):
                            self.mgm_logger('debug', open_trade_unclogger,
                                            'No unclogging needed! Currently checked pair ('
                                            + pair + ') has not been open been open for long enough!')
                        else:
                            self.mgm_logger('debug', open_trade_unclogger, 'Trade has been open for long enough! '
                                            + 'Proceeding to the next check!')

                            percentage_open_trades_losing = int((len(losing_open_trades) / len(all_open_trades)) * 100)
                            self.mgm_logger('debug', open_trade_unclogger, 'percentage_open_trades_losing: ' +
                                            str(percentage_open_trades_losing) + '%')
                            if percentage_open_trades_losing < \
                                    (self.sell___unclogger_percentage_open_trades_losing.value / self.precision):
                                self.mgm_logger('debug', open_trade_unclogger, 'No unclogging needed! ' +
                                                'Percentage of open trades losing needed has not been satisfied!')
                            else:
                                self.mgm_logger('debug', open_trade_unclogger, 'Percentage of open trades losing ' +
                                                'needed has been satisfied! Proceeding to the next check!')

                                self.mgm_logger('debug', open_trade_unclogger,
                                                'Fetching currently needed \'trend\' dataframe data to check how pair ('
                                                + pair + ') has been doing in during the last ' +
                                                str(self.sell___unclogger_trend_lookback_candles_window.value /
                                                    self.precision) + ' candles')

                                stored_trend_dataframe = {}
                                if is_live_or_dry_run:
                                    self.mgm_logger('debug', open_trade_unclogger,
                                                    'Fetching all needed \'trend\' trade data during Dry/Live Run')
                                    dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair,
                                                                                             timeframe=self.timeframe)

                                    for candle in range(1, int(self.sell___unclogger_trend_lookback_candles_window.value
                                                               / self.precision) + 1):
                                        stored_trend_dataframe[candle] = dataframe['trend'].iat[candle * -1]



                                else:
                                    self.mgm_logger('debug', open_trade_unclogger,
                                                    'Fetching all needed \'trend\' trade data during ' +
                                                    'BackTesting/HyperOpting')

                                    for candle in range(1, int(self.sell___unclogger_trend_lookback_candles_window.value
                                                               / self.precision) + 1):
                                        candle_time = current_time - timedelta(hours=candle)
                                        stored_trend_dataframe[candle] = \
                                            self.custom_info['trend_indicator'][pair].loc[candle_time]['trend']

                                if len(stored_trend_dataframe) < \
                                        (self.sell___unclogger_trend_lookback_candles_window.value / self.precision):
                                    self.mgm_logger('debug', open_trade_unclogger, 'No unclogging needed! ' +
                                                    'Not enough trend data stored yet!')
                                else:

                                    self.mgm_logger('debug', open_trade_unclogger, 'All needed \'trend\' trade ' +
                                                    'data (' + str(len(stored_trend_dataframe)) + ') fetched!')
                                    self.mgm_logger('debug', open_trade_unclogger,
                                                    'stored_trend_dataframe contents: ' + repr(stored_trend_dataframe))

                                    self.mgm_logger('debug', open_trade_unclogger, 'Calculating amount of ' +
                                                    'unclogger_candles_satisfied satisfied for pair: ' + pair)
                                    unclogger_candles_satisfied = 0
                                    for lookback_candle \
                                            in range(1, int(self.sell___unclogger_trend_lookback_candles_window.value /
                                                            self.precision) + 1):
                                        if self.sell___unclogger_enabled_when_downwards.value & \
                                                (stored_trend_dataframe[lookback_candle] == 'downwards'):
                                            unclogger_candles_satisfied += 1
                                        if self.sell___unclogger_enabled_when_sideways.value & \
                                                (stored_trend_dataframe[lookback_candle] == 'sideways'):
                                            unclogger_candles_satisfied += 1
                                        if self.sell___unclogger_enabled_when_upwards.value & \
                                                (stored_trend_dataframe[lookback_candle] == 'upwards'):
                                            unclogger_candles_satisfied += 1
                                    self.mgm_logger('debug', open_trade_unclogger, 'unclogger_candles_satisfied: ' +
                                                    str(unclogger_candles_satisfied) + ' for pair: ' + pair)


                                    if unclogger_candles_satisfied >= \
                                            (self.sell___unclogger_trend_lookback_candles_window.value /
                                             self.precision):
                                        self.mgm_logger('info', open_trade_unclogger, 'Unclogging losing trade...')
                                        return -0.00001  # Setting very low since 0% is seen as invalid by Freqtrade
                                    else:
                                        self.mgm_logger('info', open_trade_unclogger, 'No need to unclog open trade...')

            except Exception as e:
                self.mgm_logger('error', open_trade_unclogger,
                                'Following error has occurred in the Open Trade Unclogger:')
                self.mgm_logger('error', open_trade_unclogger, str(e))

        return self.stoploss

    def mgm_logger(self, message_type: str, code_section: str, message: str):
        """
        MoniGoMani Logger:
        ---------------------
        When passing a type and a message to this function it will log:
        - The timestamp of logging + the message_type provided + the message provided
        - To the console & To "./user_data/logs/freqtrade.log"
    
        :param message_type: The type of the message (INFO, DEBUG, WARNING, ERROR)
        :param code_section: The section in the code where the message occurred
        :param message: The log message to be displayed
        """

        if self.use_mgm_logging:
            if (self.mgm_log_levels_enabled['info'] is True) and (message_type.upper() == 'INFO'):
                logger.setLevel(logging.INFO)
                logger.info(code_section + ' - ' + message)
            elif (self.mgm_log_levels_enabled['debug'] is True) and (message_type.upper() == 'DEBUG'):
                logger.setLevel(logging.DEBUG)
                logger.debug(code_section + ' - ' + message)
            elif (self.mgm_log_levels_enabled['warning'] is True) and (message_type.upper() == 'WARNING'):
                logger.setLevel(logging.WARNING)
                logger.warning(code_section + ' - ' + message)
            elif (self.mgm_log_levels_enabled['error'] is True) and (message_type.upper() == 'ERROR'):
                logger.setLevel(logging.ERROR)
                logger.error(code_section + ' - ' + message)