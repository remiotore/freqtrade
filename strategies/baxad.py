from logging import FATAL, getLogger
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter
from typing import Dict
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

logger = getLogger(__name__)

# Constants for better readability
MAX_SLIPPAGE = -0.02
RETRY_LIMIT = 3
CANDLE_LOOKBACK = 200

class NASOSv4_SMA(IStrategy):
    INTERFACE_VERSION = 2

    # ROI table with realistic profit-taking steps
    minimal_roi = {
        "0": 0.04,
        "30": 0.03,
        "60": 0.02,
        "120": 0
    }

    stoploss = -0.15

    # Adjustable parameters
    base_nb_candles_buy = IntParameter(2, 20, default=8, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(2, 25, default=16, space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=0.984, space='buy', optimize=False)
    low_offset_2 = DecimalParameter(0.9, 0.99, default=0.942, space='buy', optimize=False)
    high_offset = DecimalParameter(0.95, 1.1, default=1.084, space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=1.401, space='sell', optimize=True)

    lookback_candles = IntParameter(1, 24, default=3, space='buy', optimize=True)
    profit_threshold = DecimalParameter(1.0, 1.03, default=1.008, space='buy', optimize=True)

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    timeframe = '15m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = CANDLE_LOOKBACK
    use_custom_stoploss = True

    def EWO(self, dataframe, ema_length=5, ema2_length=35):
        """
        Calculates the Elder's Weighted Oscillator (EWO).
        """
        ema1 = ta.SMA(dataframe, timeperiod=ema_length)
        ema2 = ta.SMA(dataframe, timeperiod=ema2_length)
        return (ema1 - ema2) / dataframe['low'] * 100

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.inf_1h) for pair in pairs]

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds indicators for 1-hour informative timeframe.
        """
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        informative_1h['ema_50_1h'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['rsi_1h'] = ta.RSI(informative_1h, timeperiod=14)
        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds indicators for the main timeframe.
        """
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['EWO'] = self.EWO(dataframe)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates indicators for both the main and informative timeframes.
        """
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        dataframe = self.normal_tf_indicators(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines buy conditions.
        """
        dataframe.loc[
            (dataframe['rsi_fast'] < 35) &
            (dataframe['EWO'] > self.ewo_high.value) &
            (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) &
            (dataframe['volume'] > 0),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Defines sell conditions.
        """
        dataframe.loc[
            (dataframe['close'] > dataframe['sma_9']) &
            (dataframe['volume'] > 0),
            'sell'
        ] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        """
        Custom stoploss logic.
        """
        thresholds = {
            'HSL': -0.15,
            'PF_1': 0.016,
            'SL_1': 0.014,
            'PF_2': 0.024,
            'SL_2': 0.022
        }
        if current_profit > thresholds['PF_2']:
            return stoploss_from_open(thresholds['SL_2'] + (current_profit - thresholds['PF_2']), current_profit)
        elif current_profit > thresholds['PF_1']:
            return stoploss_from_open(
                thresholds['SL_1'] + ((current_profit - thresholds['PF_1']) *
                (thresholds['SL_2'] - thresholds['SL_1']) / (thresholds['PF_2'] - thresholds['PF_1'])),
                current_profit
            )
        return stoploss_from_open(thresholds['HSL'], current_profit)
