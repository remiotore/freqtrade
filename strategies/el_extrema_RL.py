import numpy
import warnings
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from typing import Dict
from functools import reduce
from pandas import DataFrame, errors
from datetime import datetime
import numpy
from scipy.signal import argrelextrema
from freqtrade.strategy import (
    IStrategy, 
    DecimalParameter, 
    IntParameter, 
    CategoricalParameter, 
    BooleanParameter
)

warnings.simplefilter(action="ignore", category=errors.PerformanceWarning)




class el_extrema_RL(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True

    entry_params = {
        'base_nb_candles_entry': 12,
        'ewo_high': 4.428,
        'ewo_low': -12.383,
        'low_offset': 0.915,
        'rsi_entry': 44,
    }

    exit_params = {
        'base_nb_candles_exit': 72,
        'high_offset': 1.008,
    }

    minimal_roi = {
        '0': 0.5,
        '60': 0.45,
        '120': 0.4,
        '240': 0.3,
        '360': 0.25,
        '720': 0.2,
        '1440': 0.15,
        '2880': 0.1,
        '3600': 0.05,
        '7200': 0.02,
    }

    stoploss = -0.05
    max_open_trades = 9
    timeframe = '5m'
    informative_timeframe = '1h'
    process_only_new_candles = True
    use_custom_stoploss = False
    
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True
    
    base_nb_candles_entry = IntParameter(5, 80, default=entry_params['base_nb_candles_entry'], space='entry', optimize=True)
    base_nb_candles_exit = IntParameter(5, 80, default=exit_params['base_nb_candles_exit'], space='exit', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=entry_params['low_offset'], space='entry', optimize=True)
    high_offset = DecimalParameter(0.99, 1.1, default=exit_params['high_offset'], space='exit', optimize=True)
    
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=entry_params['ewo_low'], space='entry', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=entry_params['ewo_high'], space='entry', optimize=True)
    rsi_entry = IntParameter(30, 70, default=entry_params['rsi_entry'], space='entry', optimize=True)

    cooldown_lookback = IntParameter(2, 48, default=1, space='protection', optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space='protection', optimize=True)
    use_stop_protection = BooleanParameter(default=True, space='protection', optimize=True)

    @property
    def protections(self):
        prot = []
        prot.append(
            {
                'method': 'CooldownPeriod', 
                'stop_duration_candles': self.cooldown_lookback.value
            }
        )
        if self.use_stop_protection.value:
            prot.append(
                {
                    'method': 'StoplossGuard',
                    'lookback_period_candles': 24 * 3,
                    'trade_limit': 2,
                    'stop_duration_candles': self.stop_duration.value,
                    'only_per_pair': False,
                }
            )
        return prot
    
    @property
    def plot_config(self):
        plot_config = {}

        plot_config['main_plot'] = {
            '%-obv' : {},
            '&-s_close' : {}
        }
        plot_config['subplots'] = {
            'RSI': {
                'rsi': {}
            },
            'SMA' : {
                'sma5' : {},
                'sma35' : {},
            },
            'EWO' : {
                'EWO' : {}
            }
        }

        return plot_config
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        return dataframe
    
    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe['%-rsi-period'] = ta.RSI(dataframe, timeperiod=period)
        dataframe['%-mfi-period'] = ta.MFI(dataframe, timeperiod=period)
        dataframe['%-adx-period'] = ta.ADX(dataframe, timeperiod=period)
        dataframe['%-sma-period'] = ta.SMA(dataframe, timeperiod=period)
        dataframe['%-ema-period'] = ta.EMA(dataframe, timeperiod=period)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2.2)
        dataframe['bb_lowerband-period'] = bollinger['lower']
        dataframe['bb_middleband-period'] = bollinger['mid']
        dataframe['bb_upperband-period'] = bollinger['upper']
        dataframe['%-bb_width-period'] = (dataframe['bb_upperband-period'] - dataframe['bb_lowerband-period']) / dataframe['bb_middleband-period']
        dataframe['%-close-bb_lower-period'] = dataframe['close'] / dataframe['bb_lowerband-period']
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=period)
        dataframe['%-relative_volume-period'] = (dataframe['volume'] / dataframe['volume'].rolling(period).mean())

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        
        dataframe['%-pct-change'] = dataframe['close'].pct_change()
        dataframe['%-raw_volume'] = dataframe['volume']
        dataframe['%-raw_price'] = dataframe['close']
        dataframe['%-obv'] = ta.OBV(dataframe)
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: Dict, **kwargs
    ) -> DataFrame:
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe
    
    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        dataframe['&-s_close'] = (dataframe['close'].shift(-self.freqai_info['feature_parameters']['label_period_candles']).rolling(self.freqai_info['feature_parameters']['label_period_candles']).mean()/ dataframe['close']- 1)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        for val in self.base_nb_candles_entry.range:
            dataframe[f'ma_entry_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_exit.range:
            dataframe[f'ma_exit_{val}'] = ta.EMA(dataframe, timeperiod=val)
    
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma35'] = ta.SMA(dataframe, timeperiod=35)
        dataframe['EWO'] = (dataframe['sma5']  - dataframe['sma35']) / dataframe['close'] * 100
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['&s-extrema'] = 0
        dataframe.loc[argrelextrema(dataframe['close'].values, numpy.less, order=5)[0], '&s-extrema'] = -1
        dataframe.loc[argrelextrema(dataframe['close'].values, numpy.greater, order=5)[0], '&s-extrema'] = 1

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        entry_conditions = [
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-s_close'] > 0.01) &
                (dataframe['&s-extrema'] < 0) &
                (dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & 
                (dataframe['EWO'] > self.ewo_high.value) & 
                (dataframe['rsi'] < self.rsi_entry.value) & 
                (dataframe['volume'] > 0)
            ),
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-s_close'] > 0.01) &
                (dataframe['&s-extrema'] < 0) &
                (dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & 
                (dataframe['EWO'] < self.ewo_low.value) & 
                (dataframe['volume'] > 0)
            )
        ]

        if entry_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, entry_conditions), 'enter_long'] = 1

        exit_conditions = [
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-s_close'] < -0.01) &
                (dataframe['&s-extrema'] > 0) &
                (dataframe['close'] > dataframe[f'ma_exit_{self.base_nb_candles_exit.value}'] * self.high_offset.value) & 
                (dataframe['volume'] > 0)
            )
        ]

        if exit_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-s_close'] < 0) &
                (dataframe['&s-extrema'] > 0) &
                (dataframe['close'] > dataframe[f'ma_exit_{self.base_nb_candles_exit.value}'] * self.high_offset.value) & 
                (dataframe['volume'] > 0)
            )
        ]

        if exit_long_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_long_conditions), 'exit_long'] = 1

        exit_short_conditions = [
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-s_close'] < 0) &
                (dataframe['&s-extrema'] < 0) &
                (dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & 
                (dataframe['EWO'] > self.ewo_high.value) & 
                (dataframe['rsi'] < self.rsi_entry.value) & 
                (dataframe['volume'] > 0)
            ),
            (
                (dataframe['do_predict'] == 1) &
                (dataframe['&-s_close'] < 0) &
                (dataframe['close'] < dataframe[f'ma_entry_{self.base_nb_candles_entry.value}'] * self.low_offset.value) & 
                (dataframe['EWO'] < self.ewo_low.value) & 
                (dataframe['volume'] > 0)
            )
        ]

        if exit_short_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_short_conditions), 'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0