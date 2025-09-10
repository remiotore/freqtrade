import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair





















class CombinedBinHClucAndMADV3_2(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.021,
    }

    stoploss = -0.99 # effectively disabled.

    timeframe = '5m'
    inf_1h = '1h'

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.001 # it doesn't meant anything, just to guarantee there is a minimal profit.
    ignore_roi_if_entry_signal = True

    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025

    use_custom_stoploss = True

    process_only_new_candles = False

    startup_candle_count: int = 200

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:


        if (current_profit < 0) & (current_time - timedelta(minutes=240) > trade.open_date_utc):
            return 0.01
        return 0.99

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bb_40 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['lower'] = bb_40['lower']
        dataframe['mid'] = bb_40['mid']
        dataframe['bbdelta'] = (bb_40['mid'] - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (  # strategy BinHV45
                (dataframe['close'] > dataframe['ema_200_1h']) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                dataframe['lower'].shift().gt(0) &
                dataframe['bbdelta'].gt(dataframe['close'] * 0.031) &
                dataframe['closedelta'].gt(dataframe['close'] * 0.018) &
                dataframe['tail'].lt(dataframe['bbdelta'] * 0.233) &
                dataframe['close'].lt(dataframe['lower'].shift()) &
                dataframe['close'].le(dataframe['close'].shift()) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |
            (  # strategy ClucMay72018
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < 0.985 * dataframe['bb_lowerband']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * 20)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            |
            (  # strategy MACD Low buy
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * 0.02)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open']/100)) &
                (dataframe['volume'] < (dataframe['volume'].shift() * 4)) &
                (dataframe['close'] < (dataframe['bb_lowerband'])) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            ,
            'buy'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middleband'] * 1.01) &
                (dataframe['volume'] > 0) # Make sure Volume is not 0
            )
            ,
            'sell'
        ] = 1
        return dataframe
