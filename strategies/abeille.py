






import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class abeille(IStrategy):


    INTERFACE_VERSION = 3

    can_short: bool = False


    minimal_roi = {
         "0": 10
    }


    stoploss = -0.328






    timeframe = '1h'

    process_only_new_candles = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

  
    buy_m1 = IntParameter(1, 7, default=1)
    buy_m2 = IntParameter(1, 7, default=2)
    buy_m3 = IntParameter(1, 7, default=3)
    buy_p1 = IntParameter(7, 21, default=9)
    buy_p2 = IntParameter(7, 21, default=11)
    buy_p3 = IntParameter(7, 21, default=15)
    ema = IntParameter(1, 361, default=200,space='sell', optimize=True, load=True)

    startup_candle_count: int = 200

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
                'supertrend_1_buy_ST': {'color': 'red'},
                'supertrend_2_buy_ST': {'color': 'green'},
                'supertrend_3_buy_ST': {'color': 'blue'},
                'ema200': {'color': 'orange'},
        },
        'subplots': {
        }
    }

    def informative_pairs(self):
       
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:



        dataframe['ema200'] = ta.EMA(dataframe['close'], timeperiod=self.ema.value)

        dataframe['supertrend_1_buy'] = self.supertrend(dataframe, self.buy_m1.value, self.buy_p1.value)['STX']
        dataframe['supertrend_2_buy'] = self.supertrend(dataframe, self.buy_m2.value, self.buy_p2.value)['STX']
        dataframe['supertrend_3_buy'] = self.supertrend(dataframe, self.buy_m3.value, self.buy_p3.value)['STX']
        dataframe['supertrend_1_buy_ST'] = self.supertrend(dataframe, self.buy_m1.value, self.buy_p1.value)['ST']
        dataframe['supertrend_2_buy_ST'] = self.supertrend(dataframe, self.buy_m2.value, self.buy_p2.value)['ST']
        dataframe['supertrend_3_buy_ST'] = self.supertrend(dataframe, self.buy_m3.value, self.buy_p3.value)['ST']
       
    
        
        return dataframe
    
   
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (   
               (dataframe['supertrend_1_buy'] == 'up') &
               (dataframe['supertrend_2_buy'] == 'up') & 
               (dataframe['supertrend_3_buy'] == 'up') & 
               (dataframe['close'] > dataframe['ema200'] )& # The three indicators are 'up' for the current candle
               (dataframe['volume'] > 0) # There is at least some trading volume
            ),
            'enter_long'] = 1

        dataframe.loc[
            (   
               (dataframe['supertrend_1_buy'] == 'down') &
               (dataframe['supertrend_2_buy'] == 'down') & 
               (dataframe['supertrend_3_buy'] == 'down') & 
               (dataframe['close'] < dataframe['ema200'] )& # The three indicators are 'up' for the current candle
               (dataframe['volume'] > 0) # There is at least some trading volume
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (   
               (dataframe['close'] < dataframe['ema200'] )
            ),

            'exit_long'] = 1

        dataframe.loc[
            (   
               (dataframe['close'] > dataframe['ema200'] )
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