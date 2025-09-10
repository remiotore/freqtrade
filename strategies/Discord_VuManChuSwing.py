# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
import pandas_ta as pta
pd.options.mode.chained_assignment = None  # default='warn'
from pandas import DataFrame, Series, DatetimeIndex, merge
from functools import reduce

from freqtrade.strategy import IStrategy
from freqtrade.strategy import informative, BooleanParameter, merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
#from user_data.freqtrade3cw import Freqtrade3cw
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
import math, time

from technical.indicators import RMI, zema, ichimoku
import technical.indicators as ftt
import logging
logger = logging.getLogger(__name__)

from datetime import datetime, timedelta, timezone
from timeit import default_timer as timer

"""
	VuManChu Indicator;
	from: https://www.tradingview.com/script/q23anHmP-VuManChu-Swing-Free/
"""
#Range Size Function
def rng_size(dataframe, source='close', multiplier, period):
    df = dataframe.copy()
    wper = period * 2 - 1
    df['avrng'] = ta.EMA(math.abs(df[source] - df[source].shift()), timeperiod = period)
    df['AC'] = ta.EMA(df['avrng'], timeperiod = wper) * multiplier
    
	return df['AC']
		
#Range Filter Function
def rng_filt(dataframe, source='close', size, period):
    df = dataframe.copy()
	#size = rng_
    df['hi_band'] = np.nan
    df['lo_band'] =  np.nan
    df['rng_filt'] =  np.nan
    def calc_rng_filt1(dfr, init=0):
        global calc_rfilt_value
        global calc_src_value
        if init == 1:
            calc_rfilt_value = [0.0] * 2
            calc_src_value = [0.0] * 2
            return
        calc_src_value.pop(0)
        calc_src_value.append(dfr[source])
        calc_rfilt_value[0] = dfr[source]
        calc_rfilt_value[1] = calc_rfilt_value[0]		
        if ((calc_src_value - size) > calc_rfilt_value[1]):
            calc_rfilt_value[0] = calc_src_value - size
        if ((calc_src_value + size) < calc_rfilt_value[1]):
            calc_rfilt_value[0] = calc_src_value + size
        rng_filt1 = calc_rfilt_value[0]
        hi_band = rng_filt1 + size
        lo_band = rng_filt1 - size
        rng_filt = rng_filt1
        return hi_band, lo_band, rng_filt
    calc_rng_filt1(None, init=1)
    df[['hi_band','lo_band','rng_filt']] = df.apply(calc_rng_filt1, axis = 1, result_type='expand')
    return df[['hi_band','lo_band','rng_filt']]


class VuManChuSwing(IStrategy):

    INTERFACE_VERSION = 2
    # Buy hyperspace params:
    buy_params = {
		#VMC
        "rng_period": 20,
        "rng_multiplier": 3.5,

    }

    # Sell hyperspace params:
    sell_params = {
    }
	
	#Muss add minimum roi so that strategy work
	# ROI table:
    minimal_roi = {
 		#"0": 0.015,       
		"0": 0.01,
        #"180": 0.04, #"30"
        #"210": 0.03, #"60"
        #"300": 0.025, #"90"
    }

    # Stoploss:
    stoploss = -0.11

    # Trailing stoploss
    trailing_stop = True 
    trailing_stop_positive = 0.002  
    trailing_stop_positive_offset = 0.015 
    trailing_only_offset_is_reached = True
	
	#VMC
    optimize_buy_vmc = True
    rng_period = IntParameter(5, 600, default= int(buy_params['rng_period']), space='buy', optimize=vmc)
    rng_multiplier = DecimalParameter(0.001, 20.0, default= float(buy_params['rng_multiplier']), space='buy', optimize=optimize_buy_vmc)

		
    # Optimal timeframe for the strategy.
    timeframe = '1h'
	
    #custom_info = {}
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True


    use_custom_stoploss = True
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 300

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        #informative_pairs = [(pair, "1h") for pair in pairs]
        informative_pairs += [(pair, "4h") for pair in pairs]
        informative_pairs += [(pair, self.timeframe) for pair in pairs]
        #informative_pairs.append(('BTC/USDT', "5m"))
        informative_pairs.append(('BTC/USDT', "1h"))
        informative_pairs.append(('BTC/USDT', "4h"))
        return informative_pairs


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
	
		# VMC
        dataframe['rng_size'] = rng_size(dataframe, source='close', multiplier = self.rng_multiplier.value, period = self.rng_period.value)
//Range Filter Values
//[h_band, l_band, filt] = rng_filt(rng_src, rng_size(rng_src, rng_qty, rng_per), rng_per)		
        dataframe[['hi_band','lo_band','rng_filt']] = rng_filt(dataframe, source='close', size = dataframe['rng_size'], period = self.rng_period.value)

        return dataframe

    #@Freqtrade3cw.buy_signal
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''



//Direction Conditions
var fdir = 0.0
fdir := filt > filt[1] ? 1 : filt < filt[1] ? -1 : fdir
upward = fdir == 1 ? 1 : 0
downward = fdir == -1 ? 1 : 0

//Trading Condition
longCond = rng_src > filt and rng_src > rng_src[1] and upward > 0 or rng_src > filt and rng_src < rng_src[1] and upward > 0
shortCond = rng_src < filt and rng_src < rng_src[1] and downward > 0 or rng_src < filt and rng_src > rng_src[1] and downward > 0

CondIni = 0
CondIni := longCond ? 1 : shortCond ? -1 : CondIni[1]
longCondition = longCond and CondIni[1] == -1
shortCondition = shortCond and CondIni[1] == 1

        #Direction Conditions
        fdir = 0.0
		if dataframe['rng_filt'] > dataframe['rng_filt'].shift():
            fdir = 1
		if dataframe['rng_filt'] < dataframe['rng_filt'].shift():
            fdir = -1
		if fdir == 1:
            upward = 1
		else:
            upward = 0
		if fdir == -1:
            downward = 1
		else:
            downward = 0

#######################			
        vmc = (
			(dataframe['volume'] > 0)  
        )
        dataframe.loc[vmc, 'buy_tag'] += 'vmc_'
        conditions.append(vmc)

		
        dataframe.loc[
            #is_btc_safe &  # broken?
            # is_pump_safe &
            reduce(lambda x, y: x | y, conditions),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['volume'] > 0)  
            ),
            'sell'] = 0
        return dataframe
    