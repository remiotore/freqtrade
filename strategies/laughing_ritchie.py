from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtp
import talib.abstract as ta
from freqtrade.strategy import merge_informative_pair, informative
import pandas_ta as pta





class laughing_ritchie(IStrategy):

    timeframe = '5m'
    stoploss = -0.15
    INTERFACE_VERSION = 3
    process_only_new_candles = True
    startup_candle_count = 200

    can_short = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    minimal_roi = {"0": 999}

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = False

    order_types = {
        "entry": 'limit',
        "exit": 'limit',
        "stoploss": 'market',
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99 
    }

    @property
    def protections(self):
        return  [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2
            }
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtp.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband_20_2'] = bollinger['lower']
        dataframe['bb_upperband_20_2'] = bollinger['upper']
        dataframe['bb_middleband_20_2'] = bollinger['mid']
        dataframe['bb_width_20_2'] = ((dataframe['bb_upperband_20_2'] - dataframe['bb_lowerband_20_2']) / dataframe['bb_middleband_20_2'])

        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd_12_26_9'] = macd['macd']
        dataframe['macd_signal_12_26_9'] = macd['macdsignal']
        dataframe['macd_hist_12_26_9'] = macd['macdhist']

        return dataframe

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['chop_pta_14'] = pta.chop(dataframe["high"],dataframe["low"],dataframe["close"], length=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
           (
                 (dataframe['volume'].rolling(15).sum() > 0) &
                 (dataframe['close'] <= dataframe['bb_lowerband_20_2']) &
                 (dataframe['rsi_14'] <= 25) &
                 (dataframe['macd_12_26_9_4h'] < 0) &
                 (dataframe['chop_pta_14_15m'] >= 30)
           ),
           ['enter_long', 'enter_tag']] = (1, 'enter_long')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
           (
                 (dataframe['close'] >= dataframe['bb_middleband_20_2'])
           ),
           ['exit_long', 'exit_tag']] = (1, 'exit_long')

        return dataframe

