from pandas import DataFrame
from freqtrade.persistence import Trade
from typing import Optional
from freqtrade.strategy import IStrategy, IntParameter
import talib.abstract as ta
from technical import qtpylib
from datetime import datetime

class Boll(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '15m'
    use_exit_signal = False
    exit_profit_only = True
    exit_profit_offset = 0.1

    minimal_roi = {"0": 0.6}
    stoploss = -0.2
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True
    order_types = {'entry': 'market', 'exit': 'market', 'stoploss': 'market', 'stoploss_on_exchange': True}

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Hitung Bollinger Bands
        dataframe[['bbl', 'bbm', 'bbu']] = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=9,
            stds=2
        )[['lower', 'mid', 'upper']]
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (dataframe['close'] < dataframe['bbu']) &  # Harga close berada di bawah Bollinger Band Upper
            (dataframe['high'] >= dataframe['bbu']), # Harga sebelumnya menyentuh atau melewati Bollinger Band Upper
            ['enter_long', 'enter_tag']
        ] = (1, 'Momentum Long')
        
        dataframe.loc[
            (dataframe['close'] > dataframe['bbl']) &
            (dataframe['low'] <= dataframe['bbl']),
            ['enter_short', 'enter_tag']
        ] = (1, 'Momentum Short')
    
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['close'] < dataframe['bbm']),
            ['exit_long', 'exit_tag']
        ] = (1, 'Trend Down')
    
        dataframe.loc[
            (dataframe['close'] > dataframe['bbm']),
            ['exit_short', 'exit_tag']
        ] = (1, 'Trend Up')

        return dataframe
