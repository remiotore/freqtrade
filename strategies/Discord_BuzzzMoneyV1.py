# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy import merge_informative_pair


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from functools import reduce
from datetime import datetime, timedelta


class BuzzzMoneyV1(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2





    buy_params = {
        "buy_dip_candles_1": 3,
        "buy_dip_candles_2": 29,
        "buy_dip_candles_3": 130,
        "buy_dip_threshold_1": 0.13,
        "buy_dip_threshold_2": 0.2,
        "buy_dip_threshold_3": 0.25,
        "base_nb_candles_buy": 19,  # value loaded from strategy
        "low_offset": 0.969,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 30,  # value loaded from strategy
        "high_offset": 1.012,  # value loaded from strategy
     }

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 10,
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.999



    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    # Custom stoploss
    use_custom_stoploss = False


    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True


    inf_1h = '1h' # informative tf


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "RSI": {
                'rsi': {'color': 'black'},
            },
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "ADOSC": {
                'ADOSC' : { 'color': 'green'},
            },
            "STOCH": {
                'slowk': { 'color': 'blue'},
                'slowd': { 'color': 'orange'},
            },

        }
    }

    optimize_dip = False
    optimize_non_dip = True
 
    base_nb_candles_buy = IntParameter(5, 80, default=30, space='buy', optimize=optimize_non_dip, load=True)
    low_offset = DecimalParameter(0.8, 0.99, default=0.958, space='buy', optimize=optimize_non_dip, load=True)
 
    buy_dip_threshold_1 = DecimalParameter(0.08, 0.2, default=0.12, space='buy', decimals=2, optimize=optimize_dip, load=True)
    buy_dip_threshold_2 = DecimalParameter(0.02, 0.5, default=0.28, space='buy', decimals=2, optimize=optimize_dip, load=True)
    buy_dip_threshold_3 = DecimalParameter(0.02, 0.5, default=0.28, space='buy', decimals=2, optimize=optimize_dip, load=True)
    buy_dip_candles_1 = IntParameter(1, 20, default=2,  space='buy', optimize=optimize_dip, load=True)
    buy_dip_candles_2 = IntParameter(1, 40, default=10, space='buy', optimize=optimize_dip, load=True)
    buy_dip_candles_3 = IntParameter(40, 140, default=132, space='buy', optimize=optimize_dip, load=True)


    # sell params
    base_nb_candles_sell = IntParameter(5, 80, default=30, load=True, optimize=True, space='sell')
    high_offset = DecimalParameter(0.8, 1.1, default=1.012, load=True, optimize=True, space='sell')

    sell_rsi_main = DecimalParameter(72.0, 90.0, default=80, space='sell', decimals=2, optimize=True, load=True)


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_1h) for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        # SMA
        # informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)
        # informative_1h['sma_200_dec'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(20)
        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        # stochastic slow
        stoch = ta.STOCH(informative_1h)
        informative_1h['slowd'] = stoch['slowd']
        informative_1h['slowk'] = stoch['slowk']
        # SSL Channels
        #ssl_down_1h, ssl_up_1h = SSLChannels(informative_1h, 20)
        #informative_1h['ssl_down'] = ssl_down_1h
        #informative_1h['ssl_up'] = ssl_up_1h
        #dataframe['ADOSC'] = ta.ADOSC(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])

        return informative_1h



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
        
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)



        # Momentum Indicators
        # ------------------------------------

        # ADX
        #dataframe['adx'] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        dataframe['lame_ao'] = qtpylib.awesome_oscillator(dataframe, weighted=False, fast=34, slow=5)

        # # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upperband"] = keltner["upper"]
        # dataframe["kc_lowerband"] = keltner["lower"]
        # dataframe["kc_middleband"] = keltner["mid"]
        # dataframe["kc_percent"] = (
        #     (dataframe["close"] - dataframe["kc_lowerband"]) /
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        # )
        # dataframe["kc_width"] = (
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        # )

        # # Ultimate Oscillator
        # dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        #stoch_fast = ta.STOCHF(dataframe)
        #dataframe['fastd'] = stoch_fast['fastd']
        #dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        # stoch_rsi = ta.STOCHRSI(dataframe)
        # dataframe['fastd_rsi'] = stoch_rsi['fastd']
        # dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        macd = ta.MACD(dataframe, fastperiod=8, slowperiod=21, signalperiod=5)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        #dataframe['mfi'] = ta.MFI(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        # dataframe["bb_percent"] = (
        #     (dataframe["close"] - dataframe["bb_lowerband"]) /
        #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        # )
        # dataframe["bb_width"] = (
        #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        # )

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) / dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe[f'ema_{self.base_nb_candles_sell.value}'] = ta.EMA(dataframe, timeperiod=int(self.base_nb_candles_sell.value))
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # # SMA - Simple Moving Average
        dataframe[f'sma_{self.base_nb_candles_buy.value}'] = ta.SMA(dataframe, timeperiod=int(self.base_nb_candles_buy.value))
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        # dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)
        #dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        #dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)

        # Parabolic SAR
        #dataframe['sar'] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        #dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        #hilbert = ta.HT_SINE(dataframe)
        #dataframe['htsine'] = hilbert['sine']
        #dataframe['htleadsine'] = hilbert['leadsine']

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']


        #Chaikin A/D Oscillator
        dataframe['ADOSC'] = ta.ADOSC(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """


        dataframe.loc[
            (
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (((dataframe['open'].rolling(int(self.buy_dip_candles_1.value)).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                (((dataframe['open'].rolling(int(self.buy_dip_candles_1.value + self.buy_dip_candles_2.value)).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &
                (((dataframe['open'].rolling(int(self.buy_dip_candles_1.value + self.buy_dip_candles_2.value + self.buy_dip_candles_3.value)).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_3.value) &

                #(dataframe['ADOSC'] > 0) &
                #(dataframe['rsi'] > 50) &
                #(dataframe['rsi'].shift() > 50) &
                #(dataframe['rsi'].shift(2) > 50) &
                #(dataframe['rsi_1h'] > 50) &
                #(dataframe['slowk'] > dataframe['slowd']) &
                #(dataframe['slowk'] > 80) &
                #(dataframe['slowk_1h'] > dataframe['slowd_1h']) &
                #(dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['lame_ao'] > dataframe['lame_ao'].shift()) &
                (dataframe['lame_ao'].shift() < dataframe['lame_ao'].shift(2)) &
                
                #(qtpylib.crossed_above(dataframe['ADOSC'], 0)) &
                #(qtpylib.crossed_above(dataframe['ao'], 0)) &
                #(qtpylib.crossed_above(dataframe['rsi'], 50)) &  # Signal: RSI crosses above 50
                #(qtpylib.crossed_above(dataframe['slowk'], dataframe['slowd'])) &  # stochastic k crosses above d
                #(qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &  # stochastic macd crosses above macdsignals
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe['lame_ao'] < dataframe['lame_ao'].shift()) &
                (dataframe['lame_ao'].shift() > dataframe['lame_ao'].shift(2)) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe

