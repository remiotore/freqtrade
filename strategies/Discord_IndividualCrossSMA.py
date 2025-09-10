from os import closerange
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

"""
"BTC/BUSD",
"ETH/BUSD",
"BNB/BUSD",
"SOL/BUSD",
"XRP/BUSD",
"ADA/BUSD",
"AVAX/BUSD",
"DOT/BUSD",
"DOGE/BUSD",
"SHIB/BUSD",
"MATIC/BUSD",
"NEAR/BUSD",
"ATOM/BUSD",
"LTC/BUSD",
"LINK/BUSD",
"UNI/BUSD",
"TRX/BUSD",
"BCH/BUSD",
"FTT/BUSD",
"ETC/BUSD",
"ALGO/BUSD",
"XLM/BUSD",
"VET/BUSD",
"MANA/BUSD",
"SAND/BUSD",
"AXS/BUSD",
"ZIL/BUSD",
"XMR/BUSD",
"CAKE/BUSD",
"RVN/BUSD",
"ICP/BUSD",
"XNO/BUSD",
"SLP/BUSD",
"KDA/BUSD",
"FLUX/BUSD",
"AAVE/BUSD",
StopLoss -0.05
Max Open Trades 36
"""


class IndividualCrossSMA(IStrategy):    
    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 100
    }

    stoploss = -0.05
    #trailing_stop = True
    #trailing_only_offset_is_reached = True
    #trailing_stop_positive = 0.025
    #trailing_stop_positive_offset = 0.05  # Disabled / not configured

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    process_only_new_candles = False
    startup_candle_count: int = 100

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'emergency_exit': 'market'
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def informative_pairs(self):
        return []
    
    buy_params = {
        "buy_sma_short_btc": 57,
        "buy_sma_short_eth": 63,
        "buy_sma_short_bnb": 73,
        "buy_sma_short_sol": 91,
        "buy_sma_short_xrp": 71,
        "buy_sma_short_ada": 77,
        "buy_sma_short_avax": 53,
        "buy_sma_short_dot": 71,
        "buy_sma_short_doge": 67,
        "buy_sma_short_shib": 97,
        "buy_sma_short_matic": 57,
        "buy_sma_short_near": 63,
        "buy_sma_short_atom": 65,
        "buy_sma_short_ltc": 37,
        "buy_sma_short_link": 59,
        "buy_sma_short_uni": 57,
        "buy_sma_short_trx": 93,
        "buy_sma_short_bch": 67,
        "buy_sma_short_ftt": 59,
        "buy_sma_short_etc": 93,
        "buy_sma_short_algo": 43,
        "buy_sma_short_xlm": 65,
        "buy_sma_short_vet": 61,
        "buy_sma_short_mana": 15,
        "buy_sma_short_sand": 63,
        "buy_sma_short_axs": 55,
        "buy_sma_short_zil": 21,
        "buy_sma_short_xmr": 81,
        "buy_sma_short_cake": 45,
        "buy_sma_short_rvn": 63,
        "buy_sma_short_icp": 85,
        "buy_sma_short_xno": 95,
        "buy_sma_short_slp": 11,
        "buy_sma_short_kda": 35,
        "buy_sma_short_flux": 83,
        "buy_sma_short_aave": 65
    }
    sell_params = {     
        "sell_sma_long_btc": 47,
        "sell_sma_long_eth": 51,
        "sell_sma_long_bnb": 95,
        "sell_sma_long_sol": 99,
        "sell_sma_long_xrp": 87,
        "sell_sma_long_ada": 97,
        "sell_sma_long_avax": 25,
        "sell_sma_long_dot": 55,
        "sell_sma_long_doge": 77,
        "sell_sma_long_shib": 99,
        "sell_sma_long_matic": 39,
        "sell_sma_long_near": 53,
        "sell_sma_long_atom": 57,
        "sell_sma_long_ltc": 23,
        "sell_sma_long_link": 43,
        "sell_sma_long_uni": 39,
        "sell_sma_long_trx": 95,
        "sell_sma_long_bch": 11,
        "sell_sma_long_ftt": 31,
        "sell_sma_long_etc": 97,
        "sell_sma_long_algo": 27,
        "sell_sma_long_xlm": 39,
        "sell_sma_long_vet": 53,
        "sell_sma_long_mana": 25,
        "sell_sma_long_sand": 27,
        "sell_sma_long_axs": 47,
        "sell_sma_long_zil": 89,
        "sell_sma_long_xmr": 71,
        "sell_sma_long_cake": 79,
        "sell_sma_long_rvn": 75,
        "sell_sma_long_icp": 47,
        "sell_sma_long_xno": 83,
        "sell_sma_long_slp": 25,
        "sell_sma_long_kda": 7,
        "sell_sma_long_flux": 91,
        "sell_sma_long_aave": 55
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        par_minuscula = metadata['pair'].replace('/BUSD', '').lower()
        compra = f'buy_sma_short_{par_minuscula}'
        venta = f'sell_sma_long_{par_minuscula}'
        #Creo Short SMAs de compra y venta
        dataframe = pd.concat([dataframe, pd.DataFrame(ta.SMA(dataframe, timeperiod=self.buy_params[compra]), columns=[f'{compra}_{self.buy_params[compra]}'])], axis=1)
        dataframe = pd.concat([dataframe, pd.DataFrame(ta.SMA(dataframe, timeperiod=self.sell_params[venta]), columns=[f'{venta}_{self.sell_params[venta]}'])], axis=1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        par_minuscula = metadata['pair'].replace('/BUSD', '').lower()
        compra = f'buy_sma_short_{par_minuscula}'
        venta = f'sell_sma_long_{par_minuscula}'
        conditions.append(
        qtpylib.crossed_above(dataframe[f'{compra}_{self.buy_params[compra]}'], dataframe[f'{venta}_{self.sell_params[venta]}']))
        conditions.append(dataframe['volume'] > 0)  

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        par_minuscula = metadata['pair'].replace('/BUSD', '').lower()
        compra = f'buy_sma_short_{par_minuscula}'
        venta = f'sell_sma_long_{par_minuscula}'
        conditions.append(
        qtpylib.crossed_below(dataframe[f'{compra}_{self.buy_params[compra]}'], dataframe[f'{venta}_{self.sell_params[venta]}']))
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe
    