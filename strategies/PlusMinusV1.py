import datetime
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from technical import qtpylib
from freqtrade.strategy import IStrategy, IntParameter


class PlusMinusV1(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    can_short = True
    use_exit_signal=True
    exit_profit_only = True
    exit_profit_offset = 0.5

    buy_params = {
        "window": 128,
    }

    minimal_roi = {}

    stoploss = -0.3

    trailing_stop = False

    max_open_trades = -1

    window = IntParameter(1, 1000, default=buy_params['window'], space="buy")

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {},
            'subplots' : {
                'RSI' : {
                    'RSI' : {},
                    'RSI Max'  : { 'plotly': { 'mode': 'markers', 'marker': {'size': 8, 'line': { 'width': 2 }, 'color': 'green' } } },
                    'RSI Min'  : { 'plotly': { 'mode': 'markers', 'marker': {'size': 8, 'line': { 'width': 2 }, 'color': 'red' } } },
                },
                'DI' : {
                    'Plus DI'      : { 'color' : 'green'  },
                    'Plus DI Max'  : { 'plotly': { 'mode': 'markers', 'marker': {'size': 8, 'line': { 'width': 2 }, 'color': 'green' } } },
                    'Plus DI Min'  : { 'plotly': { 'mode': 'markers', 'marker': {'size': 8, 'line': { 'width': 2 }, 'color': 'red' } } },
                    'Minus DI'     : { 'color' : 'blue' },
                    'Minus DI Max' : { 'plotly': { 'mode': 'markers', 'marker': {'size': 8, 'line': { 'width': 2 }, 'color': 'green' } } },
                    'Minus DI Min' : { 'plotly': { 'mode': 'markers', 'marker': {'size': 8, 'line': { 'width': 2 }, 'color': 'red' } } },
                },
            },
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['RSI'] = ta.RSI(dataframe)
        dataframe['RSI Max'] = dataframe['RSI'].rolling(self.window.value).max().where(dataframe['RSI'] == dataframe['RSI'].rolling(self.window.value).max(), np.nan)
        dataframe['RSI Min'] = dataframe['RSI'].rolling(self.window.value).min().where(dataframe['RSI'] == dataframe['RSI'].rolling(self.window.value).min(), np.nan)

        dataframe['Plus DI']  = ta.PLUS_DI(dataframe)
        dataframe['Plus DI Max'] = dataframe['Plus DI'].rolling(self.window.value).max().where(dataframe['Plus DI'] == dataframe['Plus DI'].rolling(self.window.value).max(), np.nan)
        dataframe['Plus DI Min'] = dataframe['Plus DI'].rolling(self.window.value).min().where(dataframe['Plus DI'] == dataframe['Plus DI'].rolling(self.window.value).min(), np.nan)

        dataframe['Minus DI'] = ta.MINUS_DI(dataframe)
        dataframe['Minus DI Max'] = dataframe['Minus DI'].rolling(self.window.value).max().where(dataframe['Minus DI'] == dataframe['Minus DI'].rolling(self.window.value).max(), np.nan)
        dataframe['Minus DI Min'] = dataframe['Minus DI'].rolling(self.window.value).min().where(dataframe['Minus DI'] == dataframe['Minus DI'].rolling(self.window.value).min(), np.nan)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    qtpylib.crossed_above(dataframe['Plus DI'], dataframe['Minus DI']) &
                    dataframe['RSI'] > 60
                ) | (
                    (dataframe['RSI Max'].notnull()) &
                    (dataframe['Plus DI Max'].notnull())
                )
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (
                    qtpylib.crossed_above(dataframe['Minus DI'], dataframe['Plus DI']) &
                    dataframe['RSI'] < 15
                ) |
                (
                    (dataframe['RSI Min'].notnull()) &
                    (dataframe['Minus DI Max'].notnull())
                )
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['RSI Max'].notnull()) &
                    (dataframe['Plus DI Max'].notnull())   
                ) | (
                    qtpylib.crossed_above(dataframe['Minus DI'], dataframe['Plus DI']) &
                    dataframe['RSI'] < 60
                )
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                (
                    (dataframe['RSI Min'].notnull()) &
                    (dataframe['Minus DI Max'].notnull())
                ) | (
                    qtpylib.crossed_above(dataframe['Plus DI'], dataframe['Minus DI']) &
                    dataframe['RSI'] > 15
                )
            ),
        'exit_short'] = 1

        return dataframe