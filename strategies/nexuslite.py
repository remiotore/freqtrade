from freqtrade.strategy import IStrategy
import talib
import talib.abstract as ta
import pandas as pd
import pandas_ta as pdt
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (IStrategy, DecimalParameter, IntParameter, CategoricalParameter, BooleanParameter)
from datetime import datetime
from pandas import DataFrame, errors
from functools import reduce
import numpy as np

class nexuslite(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    
    stoploss = -0.05
    timeframe = '15m'
    process_only_new_candles = True
    use_custom_stoploss = False
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True
    
    cooldown_lookback = IntParameter(2, 48, default=5, space='protection', optimize=True)
    stop_duration = IntParameter(12, 200, default=178, space='protection', optimize=True)
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
            'close': {}
        }
        plot_config['subplots'] = {
            'RSI_BB': {
                f'rsi_{self.rsi_period.value}': {'color': 'blue'},
                f'basis_{self.ma_period.value}_{self.rsi_period.value}': {'color': 'orange'},
                f'upper_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}': {'color': 'green'},
                f'lower_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}': {'color': 'red'},
                f'disp_up_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}_{self.dispersion.value}': {'color': 'purple'},
                f'disp_down_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}_{self.dispersion.value}': {'color': 'brown'}
            },
            'impulse': {
                f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}_{self.silence.value}': {'color': 'orange'},
            },
            'microtrend': {
                f'microtrend_{self.pmom.value}_{self.nmom.value}': {'color': 'green'},
            }
        }

        return plot_config
    
    
    
    #rsi_bb_disp
    rsi_period = IntParameter(5, 50, default=25, space='buy', optimize=True)
    ma_period = IntParameter(10, 200, default=100, space='buy', optimize=True)
    stdev_multiplier = DecimalParameter(1, 5, decimals=1, default=2, space='buy', optimize=True)
    dispersion = DecimalParameter(0.01, 1.0, decimals=2, default=0.08, space='buy', optimize=True)
    
    #inpulse
    lookback = IntParameter(5, 50, default=13, space='buy', optimize=True)
    mma_length = IntParameter(5, 200, default=13, space='buy', optimize=True)
    mma_type = CategoricalParameter(["EMA", "KAMA", "TRIMA", "WMA", "DEMA", "HMA", "HWMA", "FWMA"], default="EMA", space="buy")
    smma_length = IntParameter(5, 200, default=13, space='buy', optimize=True)
    smma_type = CategoricalParameter(["EMA", "KAMA", "TRIMA", "WMA", "DEMA", "HMA", "HWMA", "FWMA"], default="EMA", space="buy")
    n_length = IntParameter(1, 10, default=3, space='buy', optimize=True)
    silence = DecimalParameter(0.01, 0.1, decimals=2, default=0.05, space='buy', optimize=True)
    strike = DecimalParameter(0.7, 0.9, decimals=2, default=0.7, space='buy', optimize=True)
    
    #microtrend
    rmi_length = IntParameter(5, 50, default=25, space='buy', optimize=True)
    pmom = IntParameter(60, 90, default=70, space='buy', optimize=True)
    nmom = IntParameter(10, 40, default=30, space='buy', optimize=True)

    #entry_conditions
    enl1_l_rsi_dd_du_u = CategoricalParameter([True, False], default=False, space="buy")
    enl2_rsi_l_dd_du_u = CategoricalParameter([True, False], default=False, space="buy")
    enl3_tdfi_str = CategoricalParameter([True, False], default=False, space="buy")
    enl4_tdfi_neg = CategoricalParameter([True, False], default=False, space="buy")
    enl5_mcr = CategoricalParameter([True, False], default=False, space="buy")
    
    ens1_l_dd_du_rsi_u = CategoricalParameter([True, False], default=False, space="sell")
    ens2_l_dd_du_u_rsi = CategoricalParameter([True, False], default=False, space="sell")
    ens3_str_tdfi = CategoricalParameter([True, False], default=False, space="sell")
    ens4_pos_tdfi = CategoricalParameter([True, False], default=False, space="sell")
    ens5_mcr = CategoricalParameter([True, False], default=False, space="sell")
    
    #exit_conditions
    exl1_l_dd_du_u_rsi = CategoricalParameter([True, False], default=False, space="buy")
    exl2_sil_tdfi = CategoricalParameter([True, False], default=False, space="buy")
    exl3_str_tdfi = CategoricalParameter([True, False], default=False, space="buy")
    exl4_mcr = CategoricalParameter([True, False], default=False, space="buy")
    
    exs1_rsi_l_dd_du_u = CategoricalParameter([True, False], default=False, space="sell")
    exs2_tdfi_sil = CategoricalParameter([True, False], default=False, space="sell")
    exs3_tdfi_str = CategoricalParameter([True, False], default=False, space="sell")
    exs4_mcr = CategoricalParameter([True, False], default=False, space="sell")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        for rsi_period in self.rsi_period.range:
            dataframe[f'rsi_{rsi_period}'] = ta.RSI(dataframe['close'], timeperiod=rsi_period)

        for ma_period in self.ma_period.range:
            for rsi_period in self.rsi_period.range:
                dataframe[f'basis_{ma_period}_{rsi_period}'] = dataframe[f'rsi_{rsi_period}'].rolling(window=ma_period).mean()
                for stdev_multiplier in self.stdev_multiplier.range:
                    dataframe[f'dev_{ma_period}_{rsi_period}_{stdev_multiplier}'] = stdev_multiplier * dataframe[f'rsi_{rsi_period}'].rolling(window=ma_period).std()
                    dataframe[f'upper_{ma_period}_{rsi_period}_{stdev_multiplier}'] = dataframe[f'basis_{ma_period}_{rsi_period}'] + dataframe[f'dev_{ma_period}_{rsi_period}_{stdev_multiplier}']
                    dataframe[f'lower_{ma_period}_{rsi_period}_{stdev_multiplier}'] = dataframe[f'basis_{ma_period}_{rsi_period}'] - dataframe[f'dev_{ma_period}_{rsi_period}_{stdev_multiplier}']
                    for dispersion in self.dispersion.range:
                        dataframe[f'disp_up_{ma_period}_{rsi_period}_{stdev_multiplier}_{dispersion}'] = dataframe[f'basis_{ma_period}_{rsi_period}'] + 2 * dataframe[f'dev_{ma_period}_{rsi_period}_{stdev_multiplier}'] * dispersion
                        dataframe[f'disp_down_{ma_period}_{rsi_period}_{stdev_multiplier}_{dispersion}'] = dataframe[f'basis_{ma_period}_{rsi_period}'] - 2 * dataframe[f'dev_{ma_period}_{rsi_period}_{stdev_multiplier}'] * dispersion

        ma_types = {
            "EMA": ta.EMA,
            "KAMA": ta.KAMA,
            "TRIMA": ta.TRIMA,
            "WMA": ta.WMA,
            "DEMA": ta.DEMA,
            "HMA": pdt.hma,
            "HWMA": pdt.hwma,
            "FWMA": pdt.fwma,
        }

        for lookback_value in self.lookback.range:
            for mma_length_value in self.mma_length.range:
                for mma_type in self.mma_type.range:
                    mma_type = ma_types[self.mma_type.value] 
                    for smma_length_value in self.smma_length.range:
                        for smma_type in self.smma_type.range:
                            smma_type = ma_types[self.smma_type.value] 
                            for n_length_value in self.n_length.range:
                                dataframe[f'mma_{mma_length_value}'] = mma_type(dataframe['close'] * 1000, timeperiod=mma_length_value)
                                dataframe[f'smma_{smma_length_value}'] = smma_type(dataframe[f'mma_{mma_length_value}'], timeperiod=smma_length_value)
                                dataframe[f'impetmma_{mma_length_value}'] = dataframe[f'mma_{mma_length_value}'] - dataframe[f'mma_{mma_length_value}'].shift(1)
                                dataframe[f'impetsmma_{smma_length_value}'] = dataframe[f'smma_{smma_length_value}'] - dataframe[f'smma_{smma_length_value}'].shift(1)
                                dataframe[f'divma_{mma_length_value}'] = np.abs(dataframe[f'mma_{mma_length_value}'] - dataframe[f'smma_{smma_length_value}'])
                                dataframe[f'averimpet_{mma_length_value}'] = (dataframe[f'impetmma_{mma_length_value}'] + dataframe[f'impetsmma_{smma_length_value}']) / 2
                                dataframe[f'tdf_{mma_length_value}_{n_length_value}'] = np.power(dataframe[f'divma_{mma_length_value}'], 1) * np.power(dataframe[f'averimpet_{mma_length_value}'], n_length_value)
                                dataframe[f'tdfi_{lookback_value}_{mma_length_value}_{n_length_value}'] = dataframe[f'tdf_{mma_length_value}_{n_length_value}'] / dataframe[f'tdf_{mma_length_value}_{n_length_value}'].rolling(window=lookback_value * n_length_value).apply(lambda x: np.max(np.abs(x)))

        for strike in self.strike.range:
            dataframe[f'strike_{strike}'] = strike
        for silence in self.silence.range:
            dataframe[f'silence_{silence}'] = silence
                    
        for rmi_length in self.rmi_length.range:
            for pmom in self.pmom.range:
                for nmom in self.nmom.range:
                    dataframe[f'up_{rmi_length}'] = ta.EMA(dataframe['close'].diff().clip(lower=0), window=rmi_length)
                    dataframe[f'down_{rmi_length}'] = ta.EMA(-dataframe['close'].diff().clip(upper=0), window=rmi_length)
                    dataframe[f'rsi_{rmi_length}'] = 100 - (100 / (1 + dataframe[f'up_{rmi_length}'] / dataframe[f'down_{rmi_length}']))
                    dataframe[f'mf_{rmi_length}'] = ta.MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], timeperiod=rmi_length)
                    dataframe[f'rsi_mfi_{rmi_length}'] = (dataframe[f'rsi_{rmi_length}'] + dataframe[f'mf_{rmi_length}']) / 2

                    # Calculate the microtrend
                    dataframe[f'ema_5_{pmom}_{nmom}'] = ta.EMA(dataframe['close'], window=5)
                    dataframe[f'ema_change_{pmom}_{nmom}'] = dataframe[f'ema_5_{pmom}_{nmom}'].diff()

                    dataframe[f'positive_mom_{pmom}_{nmom}'] = (dataframe[f'rsi_mfi_{rmi_length}'].shift(1) < pmom) & (dataframe[f'rsi_mfi_{rmi_length}'] > pmom) & (dataframe[f'rsi_mfi_{rmi_length}'] > nmom) & (dataframe[f'ema_change_{pmom}_{nmom}'] > 0)
                    dataframe[f'negative_mom_{pmom}_{nmom}'] = (dataframe[f'rsi_mfi_{rmi_length}'] < nmom) & (dataframe[f'ema_change_{pmom}_{nmom}'] < 0)               

                    dataframe[f'positive_{pmom}_{nmom}_prev'] = dataframe[f'positive_{pmom}_{nmom}'].shift(1) if f'positive_{pmom}_{nmom}' in dataframe.columns else np.nan
                    dataframe[f'negative_{pmom}_{nmom}_prev'] = dataframe[f'negative_{pmom}_{nmom}'].shift(1) if f'negative_{pmom}_{nmom}' in dataframe.columns else np.nan

                    dataframe[f'positive_{pmom}_{nmom}'] = np.where(dataframe[f'positive_mom_{pmom}_{nmom}'], True, np.where(dataframe[f'negative_mom_{pmom}_{nmom}'], False, dataframe[f'positive_{pmom}_{nmom}_prev']))
                    dataframe[f'negative_{pmom}_{nmom}'] = np.where(dataframe[f'negative_mom_{pmom}_{nmom}'], True, np.where(dataframe[f'positive_mom_{pmom}_{nmom}'], False, dataframe[f'negative_{pmom}_{nmom}_prev']))
                    
                    dataframe[f'microtrend_{pmom}_{nmom}'] = np.where(dataframe[f'positive_{pmom}_{nmom}'], 1, np.where(dataframe[f'negative_{pmom}_{nmom}'], -1, None))

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        long_conditions = [
            (dataframe[f'rsi_{self.rsi_period.value}'] < dataframe[f'upper_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}']) &
            (dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] < -dataframe[f'silence_{self.silence.value}']) &
            (dataframe['volume'] > 0)
        ]

        if self.enl1_l_rsi_dd_du_u.value:
            long_conditions.append(dataframe[f'rsi_{self.rsi_period.value}'] < dataframe[f'disp_down_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}_{self.dispersion.value}'])
        if self.enl2_rsi_l_dd_du_u.value:
            long_conditions.append(dataframe[f'rsi_{self.rsi_period.value}'] < dataframe[f'lower_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}'])
            
        if self.enl3_tdfi_str.value:
            long_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] < -dataframe[f'strike_{self.strike.value}'])
        if self.enl4_tdfi_neg.value:
            long_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] == -1)

        if self.enl5_mcr.value:
            long_conditions.append(dataframe[f'microtrend_{self.pmom.value}_{self.nmom.value}'] < 0)

        if long_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, long_conditions), 'enter_long'] = 1

        short_conditions = [
            (dataframe[f'rsi_{self.rsi_period.value}'] > dataframe[f'lower_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}']) &
            (dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] > dataframe[f'silence_{self.silence.value}']) &
            (dataframe['volume'] > 0)
        ]

        if self.ens1_l_dd_du_rsi_u.value:
            short_conditions.append(dataframe[f'rsi_{self.rsi_period.value}'] > dataframe[f'disp_up_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}_{self.dispersion.value}'])
        if self.ens2_l_dd_du_u_rsi.value:
            short_conditions.append(dataframe[f'rsi_{self.rsi_period.value}'] > dataframe[f'upper_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}'])
                        
        if self.ens3_str_tdfi.value:
            short_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] > dataframe[f'strike_{self.strike.value}'])
        if self.ens4_pos_tdfi.value:
            short_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] == 1)

        if self.ens5_mcr.value:
            short_conditions.append(dataframe[f'microtrend_{self.pmom.value}_{self.nmom.value}'] > 0)

        if short_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, short_conditions), 'enter_short'] = 1

        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        long_exit_conditions = [
            (dataframe[f'rsi_{self.rsi_period.value}'] > dataframe[f'disp_up_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}_{self.dispersion.value}']) &
            (dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] > -dataframe[f'silence_{self.silence.value}']) &
            (dataframe['volume'] > 0)
        ]

        if self.exl1_l_dd_du_u_rsi.value:
            long_exit_conditions.append(dataframe[f'rsi_{self.rsi_period.value}'] > dataframe[f'upper_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}'])
        if self.exl2_sil_tdfi.value:
            long_exit_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] > dataframe[f'silence_{self.silence.value}'])
        if self.exl3_str_tdfi.value:
            long_exit_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] > dataframe[f'strike_{self.strike.value}'])
        if self.exl4_mcr.value:
            long_exit_conditions.append(dataframe[f'microtrend_{self.pmom.value}_{self.nmom.value}'] > 0)

        if long_exit_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, long_exit_conditions), 'exit_long'] = 1

        short_exit_conditions = [
            (dataframe[f'rsi_{self.rsi_period.value}'] < dataframe[f'disp_down_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}_{self.dispersion.value}']) &
            (dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] < dataframe[f'silence_{self.silence.value}']) &
            (dataframe['volume'] > 0)
        ]

        if self.exs1_rsi_l_dd_du_u.value:
            short_exit_conditions.append(dataframe[f'rsi_{self.rsi_period.value}'] < dataframe[f'lower_{self.ma_period.value}_{self.rsi_period.value}_{self.stdev_multiplier.value}'])
        
        if self.exs2_tdfi_sil.value:
            short_exit_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] < -dataframe[f'silence_{self.silence.value}'])
        if self.exs3_tdfi_str.value:
            short_exit_conditions.append(dataframe[f'tdfi_{self.lookback.value}_{self.mma_length.value}_{self.n_length.value}'] < -dataframe[f'strike_{self.strike.value}'])

        if self.exs4_mcr.value:
            short_exit_conditions.append(dataframe[f'microtrend_{self.pmom.value}_{self.nmom.value}'] < 0)

        if short_exit_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, short_exit_conditions), 'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0