






import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class koala(IStrategy):


    INTERFACE_VERSION = 3

    can_short: bool = False


    minimal_roi = {
        "0": 1



    }


    stoploss = -0.348






    timeframe = '1h'

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    emalow = IntParameter(low=1, high=55, default=20, space='buy', optimize=True, load=True)
    emahigh = IntParameter(low=10, high=200, default=55, space='buy', optimize=True, load=True)
    emalong = IntParameter(low=55, high=361, default=200, space='buy', optimize=True, load=True)
    supertrend_m = IntParameter(low=5, high=20, default=3, space='buy', optimize=True, load=True)
    supertrend_p = IntParameter(low=5, high=100, default=10, space='buy', optimize=True, load=True)
    supertrend_m_sell = IntParameter(low=5, high=20, default=supertrend_m.value, space='sell', optimize=True, load=True)
    supertrend_p_sell = IntParameter(low=5, high=100, default=supertrend_p.value, space='sell', optimize=True, load=True)

    startup_candle_count: int = 55

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
                'emalow': {'color': 'red'},
                'emahigh': {'color': 'green'},
                'emalong': {'color': 'blue'},
                'supertrend_1_ST': {'color': 'orange'}
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'orange'},
            }
        }
    }

    def informative_pairs(self):
       
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe['trix9'] = ta.TRIX(dataframe['close'], timeperiod=9)
        dataframe['trix15'] = ta.TRIX(dataframe['close'], timeperiod=21)

        dataframe['emalow'] = ta.EMA(dataframe['close'], timeperiod=self.emalow.value)
        dataframe['emahigh'] = ta.EMA(dataframe['close'], timeperiod=self.emahigh.value)
        dataframe['emalong'] = ta.EMA(dataframe['close'], timeperiod=self.emalong.value)

        dataframe['ma361'] = ta.MA(dataframe['close'], timeperiod=361)

        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=55)

        dataframe['supertrend_1'] = self.supertrend(dataframe, self.supertrend_m.value, self.supertrend_p.value)['STX']
        dataframe['supertrend_1_ST'] = self.supertrend(dataframe, self.supertrend_m.value, self.supertrend_p.value)['ST']
        

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (   
                (dataframe['emalow'] > dataframe['supertrend_1_ST']) &
                (dataframe['supertrend_1'] == 'up') &
                (dataframe['close'] > dataframe['emalow'])

            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['emalow'] < dataframe['supertrend_1_ST']) &
                (dataframe['supertrend_1'] == 'down') &
                (dataframe['close'] < dataframe['emalow'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (   
                (dataframe['supertrend_1'] == 'down')  
            ),

            'exit_long'] = 1

        dataframe.loc[
            (   
                (dataframe['supertrend_1'] == 'up')  
            ),

            'exit_short'] = 1

        return dataframe


    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00

        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST' : df[st],
            'STX' : df[stx]
        })