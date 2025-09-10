
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta
import numpy as np


class powerx_tpsl(IStrategy):




    stoploss = -1

    timeframe = "1d"

    minimal_roi = {"0": 100.}

    order_types = {
        "buy": "limit",
        "sell": "limit",
        "emergencysell": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    plot_config = {

        "main_plot": {
            "sma": {},
        },
        "subplots": {

            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "STOCH": {
                "stochd": {"color": "red"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['rsi'] = pta.rsi(close=dataframe['close'], length=14)

        k=14
        d=3
        smooth_k=3
        stoch = pta.stoch(high=dataframe['high'],low=dataframe['low'],close=dataframe['close'], k=k, d=d, smooth_k=smooth_k)
        dataframe['stochk']=stoch[f'STOCHk_{k}_{d}_{smooth_k}']
        dataframe['stochd']=stoch[f'STOCHd_{k}_{d}_{smooth_k}']

        f=12
        s=26
        sig=9
        macd = pta.macd(close=dataframe['close'], fast=f, slow=s, signal=sig)
        dataframe['macd'] = macd[f'MACD_{f}_{s}_{sig}']
        dataframe['macd_signal'] = macd[f'MACDs_{f}_{s}_{sig}']
        dataframe['macd_hist'] = macd[f'MACDh_{f}_{s}_{sig}']

        dataframe['atr'] = pta.atr(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=14)



        def buy_sell(data):
            rsi_level = 50
            stochd_level = 50
            signal = []
            for i in range(len(dataframe)):

                if (dataframe['rsi'][i] > rsi_level) & (dataframe['stochd'][i] > stochd_level) & (dataframe['macd'][i] > dataframe['macd_signal'][i]):
                    signal.append('buy')
                else:
                    signal.append('sell')

            return signal

        dataframe['signal'] = buy_sell(dataframe)

        dataframe['advice_changed'] = dataframe['signal'].shift(+1) != dataframe['signal']

        atr_sl_mult = 1
        atr_tp_mult = 2

        for i, row in dataframe.iterrows():
            if row['advice_changed'] == True and row['signal'] == 'buy':

                dataframe.loc[i,'takeprofit'] = row['close'] + (row['atr'] * atr_tp_mult)
                dataframe.loc[i,'stoploss'] = row['close'] - (row['atr'] * atr_sl_mult)
                dataframe.loc[i,'buyprice'] = row['close']
            elif row['advice_changed'] == True and row['signal'] == 'sell':

                dataframe.loc[i,'takeprofit'] = np.nan
                dataframe.loc[i,'stoploss'] = np.nan
                dataframe.loc[i,'buyprice'] = np.nan
            else:

                dataframe.loc[i,'takeprofit'] = dataframe.loc[i-1,'takeprofit']
                dataframe.loc[i,'stoploss'] = dataframe.loc[i-1,'stoploss']
                dataframe.loc[i,'buyprice'] = dataframe.loc[i-1,'buyprice']
        
        print(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['signal'] == 'buy')
                & (dataframe['advice_changed'] == True)
            ),
            "buy",
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['signal'] == 'sell')
                & (dataframe['advice_changed'] == True)
            ),
            "sell",
        ] = 1


        return dataframe
