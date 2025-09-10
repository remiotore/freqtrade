



import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class FRAYSTRAT_BTCUSDT_1H(IStrategy):
    """
    This is a sample strategy to inspire you.
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


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 0.4,
        "32": 0.152,
        "208": 0.19,
        "525": 0.2
    }


    stoploss = -0.229

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03  # Disabled / not configured

    buy_rsi = IntParameter(low=10, high=50, default=15, space='buy', optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)
    cooldown_lookback = IntParameter(2, 90, default=3, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=3, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    @property
    def protections(self):
            return [
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": self.cooldown_lookback.value
                },
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": self.cooldown_lookback.value,
                    "trade_limit": 5,
                    "stop_duration_candles": self.stop_duration.value,
                    "max_allowed_drawdown": 0.7
                },
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 20,
                    "trade_limit": 3,
                    "stop_duration_candles": 4,
                    "only_per_pair": False
                },
                {
                    "method": "LowProfitPairs",
                    "lookback_period_candles": 24,
                    "trade_limit": 2,
                    "stop_duration_candles": 4,
                    "required_profit": 0.01
                }
                
            ]

    timeframe = '15m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 8

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
            'sar': {'color': 'blue'},
            'ema7':{'color': 'red'},
            'ema12':{'color': 'yellow'}
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
                'macdhist':{}
            },



            "FISHERS RSI":{
                'fisher_rsi':{'color':'green'},
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

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)







        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']







        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['mfi'] = ta.MFI(dataframe)





        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=18, stds=2)
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
















        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema30'] = ta.EMA(dataframe, timeperiod=30)
        dataframe['ema12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)








        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=7)



        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']




















































        """

        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

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

                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &
                (dataframe['close'] < dataframe['bb_middleband']) &  # Guard: tema below BB middle
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
                (dataframe['tema'] < dataframe['ema7']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
             )|
            
            (
                (dataframe['ema7'] < dataframe['ema12']) &
                (dataframe['rsi'] > 51 ) &
                (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                (dataframe['macdsignal'] > dataframe['macd']) &
                (dataframe['volume'] > 0)
                
            )|

            (
                (dataframe['macdsignal'] < dataframe['macd']) &
                (dataframe['ema12'] < dataframe['ema7']) &
                (dataframe['tema'] > dataframe['tema'].shift(1)) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        dataframe.loc[
            (

                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &
                (dataframe['close'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                (dataframe['tema'] < dataframe['tema'].shift(1)) & # Guard: tema is falling
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            )|
            (
                (dataframe['ema7'] > dataframe['ema12']) &
                (dataframe['tema'] > dataframe['ema7']) &
                (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
                (dataframe['macdsignal'] > dataframe['macd']) &
                (dataframe['volume'] > 0)
                 
             )|
            (
                  
                (dataframe['macdsignal'] < dataframe['macd']) &
                (dataframe['tema'] > dataframe['ema7']) &
                (dataframe['tema'] < dataframe['tema'].shift(2)) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
