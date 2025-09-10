# --- Do not remove these libs ---
from email.policy import default
from idna import valid_contextj
from freqtrade.strategy.interface import IStrategy # type: ignore
from typing import Dict, Optional, Union
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
import pandas as pd
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import logging
from datetime import datetime, timedelta
import helper_functions as helper


logger = logging.getLogger(__name__)


def calculate_chop(dataframe, period=14):
    df = dataframe.copy()
    high_n = df['high'].rolling(window=period).max()
    low_n = df['low'].rolling(window=period).min()  
    # محاسبه محدوده واقعی (TR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # مجموع محدوده واقعی (Sum TR) برای دوره n
    sum_tr = tr.rolling(window=period).sum()
    # محاسبه CHOP
    chop = 100 * np.log10(sum_tr / (high_n - low_n)) / np.log10(period)
    # مقادیر پایین CHOP (زیر 38.2) نشان‌دهنده بازاری با روند قوی است.
    # مقادیر بالای CHOP (بالای 61.8) نشان‌دهنده بازاری خنثی (بی‌روند) و پرنوسان است.
    return chop

# محاسبه‌ی میانگین وزنی محدوده (RWMA)
def range_ma(range_vals, price_vals, period):
    weights = range_vals / range_vals.rolling(window=period).sum()
    weighted_sum = (price_vals * weights).rolling(window=period).sum()
    total_weights = weights.rolling(window=period).sum()
    return weighted_sum / total_weights

def calculate_vsi(dataframe): # volume sensitivity index
    dataframe = dataframe.copy()
    # محاسبه تغییرات درصدی قیمت
    dataframe['price_change'] = dataframe['close'].pct_change()

    # محاسبه تغییرات درصدی حجم
    dataframe['volume_change'] = dataframe['volume'].pct_change()

    # محاسبه حساسیت به حجم
    dataframe['vsi'] = dataframe['price_change'] / dataframe['volume_change']
    
    # نرمال‌سازی VSI
    dataframe['vsi_normalized'] = (dataframe['vsi'] - dataframe['vsi'].min()) / (dataframe['vsi'].max() - dataframe['vsi'].min())
    
    return dataframe['vsi_normalized']


# def ConDi(dataframe, dataframe1, dataframe2):
#     # ابتدا ستون ConDi رو با مقدار پیش‌فرض 0 ایجاد می‌کنیم
#     dataframe['ConDi'] = 0
    
#     # شرط برای همگرایی (هر دو به سمت بالا یا هر دو به سمت پایین)
#     dataframe.loc[
#         ((dataframe1 > dataframe1.shift(1)) & (dataframe2 > dataframe2.shift(1))) |
#         ((dataframe1 < dataframe1.shift(1)) & (dataframe2 < dataframe2.shift(1))),
#         'ConDi'
#     ] = 1
    
#     # شرط برای واگرایی (یکی به سمت بالا و دیگری به سمت پایین)
#     dataframe.loc[
#         ((dataframe1 > dataframe1.shift(1)) & (dataframe2 < dataframe2.shift(1))) |
#         ((dataframe1 < dataframe1.shift(1)) & (dataframe2 > dataframe2.shift(1))),
#         'ConDi'
#     ] = -1
    
#     return dataframe['ConDi']

def ConDi(dataframe, dataframe1, dataframe2):
    dataframe['ConDi'] = 0
    dataframe.loc[
        ((dataframe1 > dataframe1.shift(1)) & (dataframe2 > dataframe2.shift(1))) |
        ((dataframe1 < dataframe1.shift(1)) & (dataframe2 < dataframe2.shift(1))),
        'ConDi'
    ] = 1
    dataframe.loc[
        ((dataframe1 > dataframe1.shift(1)) & (dataframe2 < dataframe2.shift(1))) |
        ((dataframe1 < dataframe1.shift(1)) & (dataframe2 > dataframe2.shift(1))),
        'ConDi'
    ] = -1
    return dataframe['ConDi']

def hma_volume(dataframe, period ):
    wma_1 = dataframe['nominal'].rolling(period//2).sum()/dataframe['volume'].rolling(period//2).sum()
    wma_2 = dataframe['nominal'].rolling(period).sum()/dataframe['volume'].rolling(period).sum()
    diff = 2 * wma_1 - wma_2
    hma = diff.rolling(int(np.sqrt(period))).mean()
    return hma

def Market_Stability_index(current_price, last_halving_price=63000, next_halving_price= 179000, last_halving_date=(datetime(2024,4, 26)), next_halving_date=(datetime(2028, 4,1))):
    days_between_halvings= (next_halving_date-last_halving_date).days
    days_since_last_halving= (datetime.now()-last_halving_date).days
    real_BTC_price= last_halving_price + (next_halving_price-last_halving_price)* (days_since_last_halving/days_between_halvings)
    stability_index = ((real_BTC_price - current_price)/ current_price)*100
    return stability_index 

def cross(dataframe1, dataframe2):
    cross_up = (dataframe1 > dataframe2) & (dataframe1.shift(1) < dataframe2.shift(1))
    cross_down = (dataframe2 > dataframe1) & (dataframe2.shift(1) < dataframe1.shift(1))
    
    if cross_up.iloc[-1]:
        return 1
    elif cross_down.iloc[-1]:
        return -1
    return 0

class Klassi(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    # Stoploss:
    stoploss = -0.33
    max_open_trades = 6     # Max Open Trades
    entry_tags_limit = 2   # limit per entry tag
    max_leverage = 15.0
    wave_length = 0
    # ROI table:
    minimal_roi = {
        "0": 0.264,     #     "0": 0.219,
        "31": 0.093,    #     "24": 0.087,
        "52": 0.026,    #     "67": 0.024,
        "94": 0         #     "164": 0
    }
    
    
    custom_roi= {
        "0": 0.05,
        "60": 0
    }
    
        # Leverage configuration
    leverage_configuration = {}

    # Create custom dictionary for storing run-time data
    custom_info = {}
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)


    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 12,
        "low_offset": 0.979, #--> Original Value is 0.915
        "long_WAD": -80,
        "long_rsi": 30,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 72,
        "high_offset": 1.008,
        "short_WAD": -20,
        "short_rsi": 70
    }

    # Variables:
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=False)
    high_offset = DecimalParameter(0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=False)
    atr_period = IntParameter(5, 30, default=14, space='stoploss', optimize=False)
    atr_multiplier = DecimalParameter(1.0, 30.0, default=10.0, space='stoploss', optimize=False)
    fast_ewo = 50
    slow_ewo = 200
    short_WAD = IntParameter(-35, 0, default=sell_params['short_WAD'], space='sell', optimize= False)
    long_WAD = IntParameter(-100, -65, default= buy_params['long_WAD'], space= 'buy', optimize= False)
    short_rsi = IntParameter(51,100, default = sell_params['short_rsi'], space= 'sell', optimize= False)
    long_rsi = IntParameter(0,49, default = buy_params['long_rsi'], space= 'buy', optimize= False)
    ket_stead = DecimalParameter(0.0, 0.1 , default=0.001, space='buy', optimize=True)
    ichimoku_period1=50
    ichimoku_period2=100
    ichimoku_period3=300
    BBperiod= 250
    roc_period= 14
    pmom = 66
    nmom = 30
    

    
    #for calculation of coral trend
    ema_fast_period=50
    ema_slow_period=200
    # cci_period=IntParameter(30, 200, default=150, space= "buy", optimize=True)
    cci_threshold=IntParameter(0, 200, default=200, space = "buy", optimize=False)
    cci_period= 15
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.03 #---> 0.002 is Original Value
    trailing_stop_positive_offset = 0.11 #--> 0.055 is Original Value
    trailing_only_offset_is_reached = True
    
    # Sell signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.10
    ignore_roi_if_entry_signal = True
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe1 = '1h'
    informative_timeframe2 = '4h'
    informative_timeframe3 = '1d'
    informative_pair= 'BTC/USDT:USDT'
    process_only_new_candles = True
    startup_candle_count = 200
    plot_config = {
        "main_plot": {
            "kama": {
            "color": "#e392fe",
            "type": "line"
            },
            "ema": {
            "color": "#f2751a",
            "type": "line"
            },
            "kc_upper": {
            "color": "#5c0700",
            "type": "line"
            },
            "kc_lower": {
            "color": "#263e0f",
            "type": "line"
            },
            "ema2": {
            "color": "#ccde01"
            },
            "kama_1h": {
            "color": "#be38f3",
            "type": "line"
            },
            "kama_4h": {
            "color": "#61177c",
            "type": "line"
            },
            "EMA_1h": {
            "color": "#d29d00",
            "type": "line"
            },
            "EMA_4h": {
            "color": "#785700",
            "type": "line"
            },
            "highBand": {
            "color": "#c0c0c0",
            "type": "line"
            },
            "lowBand": {
            "color": "#aaaaaa",
            "type": "line"
            },
            "vwap": {
            "color": "#3ef4e2"
            }
        },
        "subplots": {
            "Trends": {
            "long_slope": {
                "color": "#ff2600",
                "type": "line"
            },
            "short_slope": {
                "color": "#00f900",
                "type": "line"
            },
            "long_kama_1h": {
                "color": "#00f900",
                "type": "line"
            },
            "long_kama_4h": {
                "color": "#00f900",
                "type": "line"
            },
            "short_kama_4h": {
                "color": "#ff2600",
                "type": "line"
            },
            "long_kama": {
                "color": "#00f900",
                "type": "line"
            },
            "short_kama": {
                "color": "#ff2600",
                "type": "line"
            },
            "short_ema_1h": {
                "color": "#ff2600",
                "type": "line"
            },
            "long_ema_1h": {
                "color": "#00f900",
                "type": "line"
            },
            "long_ema_4h": {
                "color": "#00f900",
                "type": "line"
            },
            "short_ema_4h": {
                "color": "#ff2600",
                "type": "line"
            },
            "short_kama_1h": {
                "color": "#ff2600",
                "type": "line"
            },
            "long_WM2": {
                "color": "#00f900",
                "type": "line"
            },
            "short_WM2": {
                "color": "#ff2600",
                "type": "line"
            },
            "short_adx": {
                "color": "#ff2600",
                "type": "line"
            },
            "long_adx": {
                "color": "#00f900",
                "type": "line"
            },
            "long_adx_1h": {
                "color": "#00f900",
                "type": "line"
            },
            "short_adx_1h": {
                "color": "#ff2600",
                "type": "line"
            },
            "long_adx_4h": {
                "color": "#00f900",
                "type": "line"
            },
            "short_adx_4h": {
                "color": "#ff2600",
                "type": "line"
            }
            },
            "WilliamR%": {
            "WAD": {
                "color": "#f1c9fe",
                "type": "line"
            },
            "WM1": {
                "color": "#d357fe",
                "type": "line"
            },
            "WM2": {
                "color": "#61177c",
                "type": "line"
            },
            "WM0": {
                "color": "#b18cfe",
                "type": "line"
            }
            },
            "POWER": {
            "kama_slope": {
                "color": "#ff4013",
                "type": "line"
            }
            },
            "ADX": {
            "ADX": {
                "color": "#ffe2d6",
                "type": "line"
            },
            "ADX_4h": {
                "color": "#ff6a00",
                "type": "line"
            },
            "ADX_1h": {
                "color": "#ffc677",
                "type": "line"
            }
            },
            "RSI": {
            "RSI_4h": {
                "color": "#d29d00",
                "type": "line"
            },
            "RSI": {
                "color": "#fefcdd",
                "type": "line"
            },
            "RSI_1h": {
                "color": "#ffd877",
                "type": "line"
            }
            },
            "ATR%": {
            "atrpct": {
                "color": "#f8fadb",
                "type": "line"
            },
            "atrpct_4h": {
                "color": "#669c35",
                "type": "line"
            },
            "atrpct_1h": {
                "color": "#b1dd8c",
                "type": "line"
            }
            },
            "v": {
            "volume_ema": {
                "color": "#00a3d7",
                "type": "line"
            }
            }
        }
        }
    use_custom_stoploss = False
    
    # For postition sizing
    # risk_per_trade = 0.02  # Risking 2% of total capital per trade
    # stop_loss_distance = abs(trade.entry_price - stop_loss)
    # position_size = (total_capital * risk_per_trade) / stop_loss_distance
    
    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 1,
                "stop_duration_candles": 288,
                "only_per_pair": True
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 12,
                "stop_duration": 144,
                "required_profit": -0.1,
                "only_per_pair": True,
            }
        ]
        
    def cleanup_cache(self):
        """ 
        Cleanup the cache with stored dataframes if the last updated time has passed (with a margin of two minutes).
        """

        outdatedkeys = []

        currentdatetime = datetime.now()
        for key in self.custom_info['cache']:
            minutes = (currentdatetime - self.custom_info['cache'][key]['datetime']).total_seconds() / 60
            
            if (minutes - 2) > timeframe_to_minutes(self.custom_info['cache'][key]['tf']):
                outdatedkeys.append(key)

                self.log(f"Removing cache storage for '{key}' because last update was {minutes} minutes ago")

        for key in outdatedkeys:
            del self.custom_info['cache'][key]
            
    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
    
        # Setup cache for dataframes
        self.custom_info['cache'] = {}

        # Setup removal of autolocks
        self.custom_info['remove-autolock'] = []
        
        self.open_trades_tags = [] # Track open trades and their entry tags with timestamps
        self.trade_exit_times = [] # Track the exit times to prevent next trades

        # Call to super first
        super().bot_start()
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """

        # Run the cleanup of cache once every five minutes
        if current_time.minute % 5 == 0 and current_time.second <= 10:
            self.cleanup_cache()

        # Check if there are pairs set for which the Auto lock should be reoved
        if len(self.custom_info['remove-autolock']) > 0:
            self.unlock_reason("Auto lock")

            # Sanity check to alert when removing the lock failed
            for pair in self.custom_info['remove-autolock']:
                lock = self.is_locked_until(pair)
                if lock:
                    self.log(
                        f"{pair} has still an active lock until {lock}, while it should have been removed!",
                        level="ERROR"
                    )

            # Clear the list
            self.custom_info['remove-autolock'].clear()

        return super().bot_loop_start(current_time)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe1) for pair in pairs] # 1h
        informative_pairs += [(pair, self.informative_timeframe2) for pair in pairs] # 4h
        informative_pairs += [(pair, self.informative_timeframe3) for pair in pairs] # 1d
        informative_pairs += [(self.informative_pair, self.timeframe)]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe1)
        return dataframe  

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:    
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe  
        
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe1)
        informative['kama'] = ta.KAMA(informative, period= 50, fast_ema= 3, slow_ema=21)
        informative.loc[(informative['kama'].shift(1) > informative['kama']),'short_kama'] = 2  
        informative.loc[(informative['kama'].shift(1) < informative['kama']),'long_kama'] = 2  
        informative['kama_slope'] = (informative['kama']/informative['kama'].shift(1)) * 1000 - 1000   # kama_slope over 2 means a little trend has been started like  a trap. use for not_buy
        informative['RSI'] = ta.RSI(informative, period=14)
        informative['rsi_ma'] =  ta.SMA(informative['RSI'] , period = 5 )
        informative['ATR'] = ta.ATR(informative, period= 14)
        informative['ADX'] = ta.ADX(informative, period= 21)
        informative.loc[(informative['ADX'].shift(1) < informative['ADX']),'long_adx'] = 6
        informative.loc[(informative['ADX'].shift(1) > informative['ADX']),'short_adx'] = 6    
        informative['EMA'] = ta.EMA(informative, period=21)
        informative.loc[(informative['EMA'].shift(1) > informative['EMA']),'short_ema'] = 2.5  
        informative.loc[(informative['EMA'].shift(1) < informative['EMA']),'long_ema'] = 2.5  
        informative['chop'] = calculate_chop(informative, 14)
        informative['nominal'] = informative['close'] * informative['volume']
        informative['HMA'] = hma_volume(informative, period= 7)
        
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe1, ffill=True)  

        informative2 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe2)  # داده‌های تایم‌فریم دوم
        informative2['kama'] = ta.KAMA(informative2, period= 50, fast_ema= 3, slow_ema=21)
        informative2.loc[(informative2['kama'].shift(1) > informative2['kama']),'short_kama'] = 3  
        informative2.loc[(informative2['kama'].shift(1) < informative2['kama']),'long_kama'] = 3 
        informative2['kama_slope'] = (informative2['kama']/informative2['kama'].shift(1)) * 1000 - 1000   # kama_slope over 2 means a little trend has been started like  a trap. use for not_buy
        informative2['RSI'] = ta.RSI(informative2, period=14)
        informative2['ATR'] = ta.ATR(informative2, period=14)
        informative2['MFI'] = ta.MFI(dataframe, timeperiod=14)
        informative2.loc[(informative2['MFI'].shift(1) > informative2['MFI']),'short_mfi'] = 1  
        informative2.loc[(informative2['MFI'].shift(1) < informative2['MFI']),'long_mfi'] = 1 
        informative2['ADX'] = ta.ADX(informative2, period=21)
        informative2.loc[(informative2['ADX'].shift(1) > informative2['ADX']),'short_adx'] = 7  
        informative2.loc[(informative2['ADX'].shift(1) < informative2['ADX']),'long_adx'] = 7 
        informative2['EMA'] = ta.EMA(informative2, period=21)
        informative2.loc[(informative2['EMA'].shift(1) > informative2['EMA']),'short_ema'] = 3.5  
        informative2.loc[(informative2['EMA'].shift(1) < informative2['EMA']),'long_ema'] = 3.5 
        informative2['WAD'] = ta.WILLR(informative2, timeperiod=50)
        informative2['wad_sma'] = ta.SMA(informative2, period = 5 )
        informative2['chop'] = calculate_chop(informative2, 14)
        informative2['nominal'] = informative2['close'] * informative2['volume']
        informative2['HMA'] = hma_volume(informative2, period= 5)
        dataframe = merge_informative_pair(dataframe, informative2, self.timeframe, self.informative_timeframe2, ffill=True)
        
        informative3 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe3)  # داده‌های تایم‌فریم دوم
        informative3['kama'] = ta.KAMA(informative3, period= 50, fast_ema= 3, slow_ema=21)
        informative3.loc[(informative3['kama'].shift(1) > informative3['kama']),'short_kama'] = 4  
        informative3.loc[(informative3['kama'].shift(1) < informative3['kama']),'long_kama'] = 4 
        informative3['kama_slope'] = (informative3['kama']/informative3['kama'].shift(1)) * 1000 - 1000   # kama_slope over 2 means a little trend has been started like  a trap. use for not_buy
        informative3['chop'] = calculate_chop(informative3, 14)
        dataframe = merge_informative_pair(dataframe, informative3, self.timeframe, self.informative_timeframe3, ffill=True)
        
        btc_price = self.dp.get_pair_dataframe(pair=self.informative_pair, timeframe=self.timeframe)
        dataframe = merge_informative_pair(dataframe, btc_price, self.timeframe, self.timeframe, ffill=True)
        
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'volume']]
        drop_columns += [f"{s}_{self.informative_timeframe1}" for s in ['date', 'open', 'high', 'low', 'volume']]
        drop_columns += [f"{s}_{self.informative_timeframe2}" for s in ['date', 'open', 'high', 'low', 'volume']]
        drop_columns += [f"{s}_{self.informative_timeframe3}" for s in ['date', 'open', 'high', 'low', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)
        # ------------ SPREAD CALCULATOR TO CALC LEVERAGE ------------------------------
        dataframe['spread'] = 100 * abs(dataframe['high'] - dataframe['low']) / dataframe['close']
        dataframe['spread_avg'] = dataframe['spread'].rolling(window=200).mean()
        dataframe['spread_avg_small'] = ta.SMA(dataframe['spread'], timeperiod = 5)
        
        # ------------ TREND INDICATORS ------------------------------------------------
        dataframe['kama'] = ta.KAMA(dataframe, window=50, pow1=4, pow2=12, fillna=True)
        dataframe['kama_'] = ta.KAMA(dataframe['close'], timeperiod=7, fastlimit=0.02, slowlimit=0.666)
        dataframe.loc[(dataframe['kama'].shift(1) > dataframe['kama']),'short_kama'] = 1  
        dataframe.loc[(dataframe['kama'].shift(1) < dataframe['kama']),'long_kama'] = 1    
        
        dataframe['ema'] = ta.SMA(dataframe, timeperiod=13)
        dataframe.loc[(dataframe['ema'].shift(1) > dataframe['ema']),'down_ema'] = 2  
        dataframe.loc[(dataframe['ema'].shift(1) < dataframe['ema']),'long_ema'] = 2    
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod = 200)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod = 5)
        dataframe['nominal'] = dataframe['close'] * dataframe['volume']
        
        dataframe['HMA'] = hma_volume(dataframe, period= 9)
        
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        vwap = (typical_price * dataframe['volume']).cumsum() / dataframe['volume'].cumsum()
        dataframe['vwap'] = vwap


        dataframe['volume_ema'] = ta.EMA(dataframe['volume'], timeperiod=20)
        dataframe.loc[
            (dataframe['vwap'] < dataframe ['vwap'].shift(1)),
            'short_vwap'] = 1.5
        dataframe.loc[
            dataframe['vwap'] > dataframe ['vwap'].shift(1),
            'long_vwap'] = 1.5
        
        # ------------------------- POWER -------------------------------
        dataframe['ADX'] = ta.ADX(dataframe, timeperiod=21)
        dataframe.loc[(dataframe['ADX'].shift(1) > dataframe['ADX']),'short_adx'] = 5  
        dataframe.loc[(dataframe['ADX'].shift(1) < dataframe['ADX']),'long_adx'] = 5    
        
                # ATR for setting dynamic stop loss and take profit
        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atrpct'] = (dataframe['ATR']/dataframe['close'])*100
        dataframe['atrpct_1h'] = (dataframe['ATR_1h']/dataframe['close'])* 100
        dataframe['atrpct_4h'] = (dataframe['ATR_4h']/dataframe['close_4h'])*100
        
        dataframe['band'] = np.minimum(dataframe['ATR'] * 0.3, dataframe['close'] * (0.3 / 100))
        
        dataframe['vwap_upper'] = dataframe['vwap'] + 1.7 * dataframe['ATR']
        dataframe['vwap_lower'] = dataframe['vwap'] - 1.7 * dataframe['ATR']
        
        dataframe['diff_kamavwap']= dataframe['kama'] - dataframe['vwap']
        dataframe['diff_kamavwap_sma'] = ta.EMA(dataframe['diff_kamavwap'], timeperiod = 7)
        dataframe['der_diff_kamavwap'] = dataframe['diff_kamavwap_sma'].diff()/dataframe['close']
        dataframe['der_diff_kamavwap_sma'] = ta.EMA(dataframe['der_diff_kamavwap'], timeperiod = 7)
        dataframe['der_diff_kamavwap_slope'] = dataframe['der_diff_kamavwap_sma']/dataframe['der_diff_kamavwap_sma'].shift(1)

        
        dataframe['distance_index'] = dataframe['diff_kamavwap_sma']/dataframe['close']
        dataframe['diff_kamavwap_slope'] = dataframe['diff_kamavwap'] / dataframe['diff_kamavwap'].shift(1)
        dataframe['diff_kamavwap_slope_sma'] = ta.SMA(dataframe['diff_kamavwap_slope'], timeperiod = 3)
        #--------------------------OSCILLATORS-----------------------------
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['RSI'], timeperiod=8)
        dataframe['L_rsi_thres'] = (dataframe['distance_index'] * 100) + 25
        dataframe['S_rsi_thres'] = (dataframe['distance_index'] * 100) + 75
        
        dataframe['MFI'] = ta.MFI(dataframe, timeperiod=14)
        
        dataframe['rsi_mfi'] = (dataframe['MFI'] + dataframe['RSI']) / 2
        
        dataframe['cci'] = ta.CCI(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=self.cci_period)  
                # محاسبه شاخص تراکم-توزیع ویلیامز (WAD)
        dataframe['WAD'] = ta.WILLR(dataframe, timeperiod=50)
        dataframe['wad_sma']= ta.SMA(dataframe['WAD'], timeperiod = 3)
        dataframe['WM0'] = ta.SMA(dataframe['WAD'], timeperiod= 15)
        dataframe['WM1'] = ta.SMA(dataframe['WAD'], timeperiod=50)
        dataframe['WM2'] = ta.SMA(dataframe['WAD'], timeperiod=200)
        
        dataframe['long_rsi_threshold'] = 25 + (50 + dataframe['WM1']) / 2
        dataframe['short_rsi_threshold'] = 75 + (50 + dataframe['WM1']) / 2
        dataframe['diff_rsi'] = dataframe['long_rsi_threshold'].diff()
        
        dataframe['chop'] = calculate_chop(dataframe, 14)
        dataframe['chop_ma'] = ta.SMA(dataframe['chop'], 21)

        #------------------Market Stability Index  and indicators based on it ----------------------------
        dataframe['MIndex']= Market_Stability_index(dataframe['close_5m'] ,last_halving_price=63000, next_halving_price= 179000, last_halving_date=(datetime(2024,4, 26)), next_halving_date=(datetime(2028, 4,1)))
        dataframe['long_wad_threshold'] = (dataframe['MIndex'] + 70) * -1
        dataframe['short_wad_threshold']  = (30 - dataframe['MIndex']  )* -1
        
        
        #----------------------------------
        # Keltner Channel Calculation
        dataframe['kc_middle'] = ta.EMA(dataframe['close'], timeperiod=20)
        dataframe['kc_upper'] = dataframe['kc_middle'] + (ta.ATR(dataframe, timeperiod=14) * 1.5)  # Modified to 1.5x ATR for crypto
        dataframe['kc_lower'] = dataframe['kc_middle'] - (ta.ATR(dataframe, timeperiod=14) * 1.5)
        
        # --------------- DERIVITIVES --------------------------
        dataframe['typical_price'] = typical_price
        dataframe['kama_slope'] = (dataframe['kama']/dataframe['kama'].shift(1)) * 1000 - 1000   # kama_slope over 2 means a little trend has been started like  a trap. use for not_buy
        
        dataframe['der1_price'] = (dataframe['kama'] / dataframe['kama'].shift(1)) * 1000 - 1000 
        dataframe.loc[dataframe['der1_price'] > 10, 'der1_price'] = 10
        dataframe.loc[dataframe['der1_price'] < - 10, 'der1_price'] = -10            
        dataframe['der1_price_sma'] = ta.SMA(dataframe['der1_price'], timeperiod = 5)
               
        dataframe['der2_price'] = (dataframe['der1_price_sma']/dataframe['der1_price_sma'].shift(1)) * 10000 - 10000
        dataframe.loc[dataframe['der2_price'] > 10, 'der2_price'] = 10
        dataframe.loc[dataframe['der2_price'] < -10, 'der2_price'] = -10
        dataframe['der2_price_sma'] = ta.SMA(dataframe['der2_price'], timeperiod = 5)
        
        
        # ------------------ Wave length calculations ----------------------
        dataframe['cross_result'] = cross(dataframe['close'], dataframe['vwap'])

        # ---------------------------------- COnvergence divergence -------------------
        dataframe['condi_rsi_kama'] = ConDi(dataframe, dataframe['rsi_ma'], dataframe['ema'])
        dataframe['condi_wad_ema'] =  ConDi(dataframe, dataframe['wad_sma'] , dataframe['ema'])
        dataframe['condi_rsi_hma'] = ConDi(dataframe, dataframe['rsi_ma_1h'], dataframe['HMA_1h'])
        dataframe['condi_wad_hma'] = ConDi(dataframe, dataframe['wad_sma_4h'], dataframe['HMA_4h'])
        dataframe['condi_ema_hma'] = ConDi(dataframe, dataframe['ema'], dataframe['HMA'])
        
        dataframe['p_mom'] = (dataframe['rsi_mfi'].shift(1) < self.pmom) & (dataframe['rsi_mfi'] > self.pmom) & (dataframe['ema5'].diff() > 0)
        dataframe['n_mom'] = (dataframe['rsi_mfi'].shift(1) > self.nmom) & (dataframe['rsi_mfi'] < self.nmom) & (dataframe['ema5'].diff() < 0)
        # dataframe['rwma'] = range_ma(dataframe['high'] - dataframe['low'], dataframe['close'], 20)
        # dataframe['rwma_band_upper'] = dataframe['rwma'] + dataframe['band']
        # dataframe['rwma_band_lower'] = dataframe['rwma'] - dataframe['band']
        dataframe['STOCH_K'], dataframe['STOCH_D'] = ta.STOCH(dataframe['high'], dataframe['low'], dataframe['close'],
                                             fastk_period=14, slowk_period=3, slowd_period=3,
                                             slowk_matype=0, slowd_matype=0)
        
        
        # ---------------------------------- BTC Dominance data -----------------------

        
        
        
        
        #_-_____NOT CLEANED BELOW

        dataframe['atf'] = 0    
        dataframe.loc[(
            (dataframe['diff_kamavwap_slope']>1 ) &
            (dataframe['diff_kamavwap_slope'].shift(1)<1)), 'atf'
        ] = 1
        dataframe.loc[(
            (dataframe['diff_kamavwap_slope']<1 ) &
            (dataframe['diff_kamavwap_slope'].shift(1)>1)), 'atf'
        ] = -1        
    
        dataframe['vwap_slope'] = (dataframe['vwap']/dataframe['vwap'].shift(1))*1000 - 1000


        
        # dataframe['kamadiff_normalized']= (dataframe['kamadiff'] - dataframe['kamadiff'].min()) / (dataframe['kamadiff'].max() - dataframe['kamadiff'].min())
        

        dataframe['is_wave'] = np.where(
            ((abs(dataframe['diff_kamavwap_slope'] - 1 ) < 0.01 )),
            0 , 1)
        
        dataframe['vsi'] = calculate_vsi(dataframe)
        
                # Calculate new indicators and conditions for entry

        # Calculate Price Volatility Index (item 2)
        dataframe['price_volatility'] = abs(dataframe['close'] - dataframe['close'].shift(1)) / dataframe['volume_ema']

        # Dynamic Slope for Entry Signals (item 3)
        dataframe['kama_vwap_slope'] = (dataframe['kama'] - dataframe['vwap']).diff() / dataframe['close']
        dataframe['kama_vwap_slope_sma'] = ta.SMA(dataframe['kama_vwap_slope'], timeperiod=7)

        # Williams Accumulation Distribution (WAD) - Strengthen Signals (item 5)
        dataframe['wad_support'] = (dataframe['WAD'] > dataframe['WM1']) # Bullish support
        dataframe['wad_resistance'] = (dataframe['WAD'] < dataframe['WM1']) # Bearish resistance
        
        # Adding Dynamic ADX Condition for Strong Trend Confirmation
        dataframe['ADX'] = ta.ADX(dataframe, timeperiod=21)
        dataframe['adx_trend'] = (dataframe['ADX'] > 25)  # Filter only strong trends

        # Bollinger Bands to filter high volatility breakouts
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']  # Bollinger Width
        dataframe['low_volatility'] = dataframe['bb_width'] < 0.1  # Low volatility indication

        # Adding ATR as a Dynamic Volatility Filter
        dataframe['atr_threshold'] = dataframe['ATR'] < dataframe['ATR'].rolling(window=20).mean()  # Low ATR indication

        # Volume Indicator for additional confirmation
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['high_volume'] = dataframe['volume'] > dataframe['volume_sma'] * 1.5  # Volume spike confirmation

        # Enhanced RSI Sensitivity
   
        dataframe['rsi_bullish'] = dataframe['RSI'] > 50  # Uptrend confirmation with RSI
        dataframe['rsi_bearish'] = dataframe['RSI'] < 50  # Downtrend confirmation with RSI
        
        dataframe.loc[
            (dataframe['der_diff_kamavwap_sma'] < dataframe['kama_vwap_slope_sma']) 
            ,'not_short'] = True
        dataframe.loc[
            (dataframe['der_diff_kamavwap_sma'] > dataframe['kama_vwap_slope_sma']) 
            ,'not_short'] = True 





        return dataframe
    
    def get_open_trades(self):
        # Fetch all open trades
        return Trade.get_open_trades()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        # --------------------------------GPT -------------------------
        
        #         Updated Long Entry Conditions
        # پیشرفته سازی سیگنال ورود Long
        # dataframe.loc[
        #     (dataframe['typical_price'] > dataframe['vwap']) &
        #     (dataframe['long_vwap'] == 1.5) &
        #     (dataframe['adx_trend']) &
        #     (dataframe['wad_support']) &
        #     (dataframe['chop'] < 68.1) &
        #     (abs(dataframe['kama_vwap_slope_sma']) > 0.002) &
        #     (dataframe['price_volatility'] < 0.1) &
        #     (dataframe['high_volume']) &
        #     (dataframe['rsi_bullish']) &
        #     (dataframe['low_volatility']),
        #     ['enter_long', 'enter_tag']] = (1, 'Advanced_kama_vwap_adx')

        # # پیشرفته سازی سیگنال ورود Short
        # dataframe.loc[
        #     (dataframe['typical_price'] < dataframe['vwap']) &
        #     (dataframe['short_vwap'] == 1.5) &
        #     (dataframe['adx_trend']) &
        #     (dataframe['wad_resistance']) &
        #     (dataframe['chop'] < 68.1) &
        #     (abs(dataframe['kama_vwap_slope_sma']) > 0.002) &
        #     (dataframe['price_volatility'] < 0.1) &
        #     (dataframe['high_volume']) &
        #     (dataframe['rsi_bearish']) &
        #     (dataframe['low_volatility']),
        #     ['enter_short', 'enter_tag']] = (1, 'Advanced_kama_vwap_adx')

        
        # ----------------------------- Based on RSI --------------------
        # dataframe.loc[(
        #     (dataframe['der_diff_kamavwap_sma'] > dataframe['kama_vwap_slope_sma'] ) &
        #     (dataframe['der_diff_kamavwap_sma'].shift(1) < dataframe['kama_vwap_slope_sma'].shift(1) ) &
        #     (dataframe['der_diff_kamavwap_sma'] > 0.002 ) & 
        #     (dataframe['short_vwap'] == 1.5) &
        #     (dataframe['STOCH_D'] > dataframe['STOCH_K']) &
        #     (dataframe['volume'] > 100 )
        # ), ['enter_short', 'enter_tag']] = (1, 'der')
        # dataframe.loc[(
        #     (dataframe['der_diff_kamavwap_sma'] < dataframe['kama_vwap_slope_sma'] ) &
        #     (dataframe['der_diff_kamavwap_sma'].shift(1) > dataframe['kama_vwap_slope_sma'].shift(1) ) &
        #     (dataframe['der_diff_kamavwap_sma'] < -0.002 ) & 
        #     (dataframe['STOCH_D'] < dataframe['STOCH_K']) &
        #     (dataframe['long_vwap'] == 1.5) &
        #     (dataframe['volume'] > 100 )
        # ), ['enter_long', 'enter_tag']] = (1, 'der')

        # --------------------------------Trend Market Entries-----------------------------------

        # ----------------Based on kama, ema WAD (wad calculated based on MIndex) ------------
        dataframe.loc[(
            (dataframe['ADX'] > 30) &  # Ensure market is trending
            (dataframe['is_wave'] == 1 ) &
            (dataframe['condi_wad_ema'] == 1 ) &
            (dataframe['condi_rsi_kama'] == 1 ) &
            (dataframe['RSI'] < dataframe['long_rsi_threshold'] ) & #30
            (dataframe['RSI_1h'] < 40 ) &
            (dataframe['wad_sma'] < dataframe['long_wad_threshold']) &
            (dataframe['close'] < dataframe['kama']) &
            (dataframe['long_vwap'] == 1.5) &
            (dataframe['volume'] > 1000)
        ), ['enter_long', 'enter_tag']] = (1, 'kemaMIx')

        dataframe.loc[(
            (dataframe['ADX'] > 30) &  # Ensure market is trending
            (dataframe['is_wave'] == 1 ) &
            (dataframe['condi_wad_ema'] == 1 ) &
            (dataframe['condi_rsi_kama'] == 1 ) &
            (dataframe['RSI'] > dataframe['short_rsi_threshold']) & #70 
            (dataframe['wad_sma'] > dataframe['short_wad_threshold']) &
            (dataframe['close'] > dataframe['kama']) &
            (dataframe['short_vwap'] == 1.5) &
            (dataframe['volume'] > 1000)
        ), ['enter_short', 'enter_tag']] = (1, 'kemaMIx')
        # -------- based on cross of EMA200 and KAMA50 ---------------
        # dataframe.loc[(
        #     # (dataframe['atrpct'] > 1) &
        #     (dataframe['ADX'] > 25) & # it was considered that adx>40 is a super strong trend, adx< 20 considered as trange market  
        #     # (dataframe['ADX_4h'] > 20) & # it was considered that adx>40 is a super strong trend, adx< 20 considered as trange market  
        #     # (dataframe['is_wave'] == 0) &  # Ensure market is trending
        #     (dataframe['kama_'] > dataframe['ema']) & # EMA and KAMA cross
        #     (dataframe['kama_'].shift(1) < dataframe['ema'].shift(1)) &
        #     (dataframe['chop_1h'] < 68.1) &
        #     (dataframe['condi_rsi_kama'] == 1 ) &
        #     (dataframe['condi_wad_ema'] == 1 ) &
        #     (dataframe['close'] < dataframe['kc_upper']) & # For safety
        #     (abs(dataframe['diff_kamavwap_slope'] - 1 ) > 0.01) &
        #     (dataframe['volume'] > 10000)
        # ), ['enter_long', 'enter_tag']] = (1, 'L_k*ema')

        # dataframe.loc[(
        #     # (dataframe['atrpct'] > 1) &
        #     (dataframe['ADX'] > 25) & # it was considered that adx>40 is a super strong trend, adx< 20 considered as trange market  
        #     # (dataframe['ADX_4h'] > 20) & # it was considered that adx>40 is a super strong trend, adx< 20 considered as trange market  
        #     # (dataframe['is_wave'] == 0 ) &  # Ensure market is trending
        #     (dataframe['kama_'] < dataframe['ema']) & # EMA and KAMA cross
        #     (dataframe['kama_'].shift(1) > dataframe['ema'].shift(1)) &
        #     (dataframe['chop_1h'] < 68.1) &
        #     (dataframe['condi_rsi_kama'] == 1 ) &
        #     (dataframe['condi_wad_ema'] == 1 ) &
        #     (dataframe['close'] > dataframe['kc_lower']) & # for safety
        #     (abs(dataframe['diff_kamavwap_slope'] - 1 ) > 0.01) &
        #     # (dataframe['short_vwap'] == 1.5) &
        #     (dataframe['volume'] > 10000)
        # ), ['enter_short', 'enter_tag']] = (1, 'S_k*ema')
    
        # ------------------------ based on ketlner, for not strong trends -----------------
        # dataframe.loc[(
        #     (dataframe['short_vwap'] == 1.5) & # ENsure Market is Trending
        #     ((dataframe['atrpct'] < 1) & (dataframe['atrpct_1h'] < 1)) &
        #     # (dataframe['vsi'] > 0.4) & 
        #     (dataframe['close'] > dataframe['kc_upper']) &  # Price above Keltner upper band (overbought)
        #     abs((dataframe['diff_kamavwap_slope'] - 1) < self.ket_stead.value ) &
        #     (dataframe['WAD'] > dataframe['short_wad_threshold']) &
        #     (dataframe['condi_rsi_kama'] == 1 ) &
        #     (dataframe['condi_wad_ema'] == 1 ) &
        #     (dataframe['chop_4h'] < 68.1) &
        #     (dataframe['volume'] > 1000)
        # ), ['enter_short', 'enter_tag']] = (1, 'ketlner')
        
        # dataframe.loc[(
        #     (dataframe['long_vwap'] == 1.5) & # ENsure Market is Trending
        #     ((dataframe['atrpct'] < 1) & (dataframe['atrpct_1h'] < 1)) &
        #     # (dataframe['vsi'] > 0.4) & 
        #     (dataframe['close'] < (dataframe['kc_lower'])) &  # Price below Keltner lower band (oversold)
        #     (dataframe['WAD'] < dataframe['long_wad_threshold']) &
        #     (dataframe['condi_rsi_kama'] == 1 ) &
        #     (dataframe['condi_wad_ema'] == 1 ) &
        #     abs((dataframe['diff_kamavwap_slope'] - 1) < self.ket_stead.value ) &
        #     (dataframe['chop_4h'] < 68.1) &
        #     (dataframe['volume'] > 10000)
        # ), ['enter_long', 'enter_tag']] = (1, 'ketlner')
        

         
        # ----------------based on Williams ------------------------------
        # dataframe.loc[(
        #     (dataframe['ATR_4h'] < 1) &
        #     (dataframe['condi_wad_ema'] == 1 ) &
        #     (dataframe['condi_rsi_kama'] == 1 ) &
        #     (dataframe['RSI'] > 75 + (50 + dataframe['WM1']) / 2) & #70
        #     ((dataframe['WAD'] > dataframe['WM0']) | (dataframe['WAD'] > -2 )) &
        #     (dataframe['WM1'] > dataframe['WM2']) &
        #     (dataframe['chop_4h'] < 68.1) &
        #     (dataframe['volume'] > 10000)
        # ), ['enter_short', 'enter_tag']] = (1, 'SW%')
        
        # dataframe.loc[(
        #     (dataframe['ATR_4h'] < 1) & 
        #     (dataframe['condi_wad_ema'] == 1 ) &
        #     (dataframe['condi_rsi_kama'] == 1 ) &
        #     (dataframe['RSI'] < 25 + (50 + dataframe['WM1']) / 2 ) & #30
        #     (dataframe['WM1'] < dataframe['WM2']) &
        #     ((dataframe['WAD'] > dataframe['WM0']) | (dataframe['WAD'] < -98 )) &
        #     (dataframe['chop_4h'] < 68.2) &
        #     (dataframe['volume'] > 10000)
        # ), ['enter_long', 'enter_tag']] = (1, 'LW%')

        # ------- based on atf ---------------------------------
        
        dataframe.loc[(
            (dataframe['atf'] == -1) &
            (dataframe['WAD'] > dataframe['short_wad_threshold']) &
            (abs(dataframe['diff_kamavwap_slope'] - 1) > 0.003) &  # Increase slope threshold slightly for stronger signals
            (dataframe['RSI'] > 70) &  # Add RSI to confirm overbought conditions for short
            (dataframe['ADX'] > 25) &  # Only enter when ADX confirms trend strength
            (dataframe['short_vwap'] == 1.5) &
            (dataframe['volume'] > 1000) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=10).mean())  # Confirm abnormal volume
        ), ['enter_short', 'enter_tag']] = (1, 'ATF')
        dataframe.loc[(
            (abs(dataframe['diff_kamavwap_slope'] - 1) > 0.003) &
            (dataframe['atf'] == 1) &
            (dataframe['WAD'] < dataframe['long_wad_threshold']) &
            (dataframe['RSI'] < 30) &  # Add RSI to confirm oversold conditions for long
            (dataframe['ADX'] > 25) &  # Only enter when ADX confirms trend strength
            (dataframe['long_vwap'] == 1.5) &
            (dataframe['volume'] > 1000) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=10).mean())  # Confirm abnormal volume
        ), ['enter_long', 'enter_tag']] = (1, 'ATF')
        
        # ----------------------------- Based on derivatives --------------------
        dataframe.loc[(
            (dataframe['der_diff_kamavwap_slope'] < 1 ) &
            (dataframe['der_diff_kamavwap_slope'].shift(1) > 1) &  
            (dataframe['der_diff_kamavwap_sma'] > 0.001) &
            (dataframe['WAD'] > dataframe['short_wad_threshold']) &
            # (dataframe['is_wave'] == 1 ) &
            (dataframe['condi_wad_ema'] == 1 ) &
            (dataframe['condi_rsi_kama'] == 1 ) &
            (dataframe['short_vwap'] == 1.5) &
            (dataframe['STOCH_D'] > dataframe['STOCH_K']) &
            (dataframe['volume'] > 1000 )
        ), ['enter_short', 'enter_tag']] = (1, 'der2')
        dataframe.loc[(
            (dataframe['der_diff_kamavwap_slope'] > 1 ) &
            (dataframe['der_diff_kamavwap_slope'].shift(1) < 1) &
            (dataframe['der_diff_kamavwap_sma'] < -0.001) &
            (dataframe['WAD'] < dataframe['long_wad_threshold']) &
            # (dataframe['is_wave'] == 1 ) &
            (dataframe['condi_wad_ema'] == 1 ) &
            (dataframe['STOCH_D'] < dataframe['STOCH_K']) &
            (dataframe['condi_rsi_kama'] == 1 ) &
            (dataframe['long_vwap'] == 1.5) &
            (dataframe['volume'] > 1000 )
        ), ['enter_long', 'enter_tag']] = (1, 'der2')
        

        
        # -------------------------------------- GPT suggest --------------------------------------------------------------        
        
        # # # Entry Conditions for Long
        # dataframe.loc[
        #     (dataframe['typical_price'] > dataframe['vwap']) &  # Market trending up
        #     (dataframe['typical_price'].shift(1) < dataframe['vwap'].shift(1)) &
        #     (dataframe['long_vwap'] == 1.5) &  # Long VWAP signal
        #     (dataframe['wad_support']) &  # WAD support
        #     (dataframe['chop'] < 68.1) &  # Low choppiness, potential trend
        #     (abs(dataframe['kama_vwap_slope_sma']) > 0.002) &  # Uptrend slope
        #     (dataframe['price_volatility'] < 0.1),  # Filter low volatility
        #     ['enter_long', 'enter_tag']] = (1, 'kama_vwap')

        # # Entry Conditions for Short
        # dataframe.loc[
        #     (dataframe['typical_price'] < dataframe['vwap']) &  # Market trending down
        #     (dataframe['typical_price'].shift(1) > dataframe['vwap'].shift(1)) &
        #     (dataframe['short_vwap'] == 1.5) &  # Short VWAP signal
        #     (dataframe['wad_resistance']) &  # WAD resistance
        #     (dataframe['chop'] < 68.1) &  # Low choppiness, potential trend
        #     (abs(dataframe['kama_vwap_slope_sma']) > 0.002) &  # Downtrend slope
        #     (dataframe['price_volatility'] < 0.1),  # Filter low volatility
        #     ['enter_short', 'enter_tag']] = (1, 'kama_vwap')
        
        
        # --------------------------- Advanced RSI -------------------------------
        
        
        
        # ----------------------------based on RSI, WAD and WAD.mean() for range markets ----------------------------------------
        dataframe.loc[(
            (dataframe['short_rsi_threshold'] > dataframe['S_rsi_thres']) &
            (dataframe['diff_rsi'] < 0) &
            (dataframe['diff_rsi'].shift(1) > 0) &
            (dataframe['close'] > dataframe['kama_']) &
            (dataframe['RSI'] > 70) &
            (dataframe['ADX'] > 25) &  # Add ADX filter for stronger trends
            (dataframe['ATR'] > dataframe['ATR'].rolling(window=14).mean()) &  # Filter for volatility
            (dataframe['short_vwap'] == 1.5) &
            (dataframe['STOCH_D'] > dataframe['STOCH_K']) &
            (dataframe['volume'] > 1000)
        ), ['enter_short', 'enter_tag']] = (1, 'rsi')
        
        dataframe.loc[(
            (dataframe['long_rsi_threshold'] < dataframe['L_rsi_thres']) &
            (dataframe['diff_rsi'] > 0) &
            (dataframe['diff_rsi'].shift(1) < 0) &
            (dataframe['close'] < dataframe['kama_']) &
            (dataframe['RSI'] < 30) &
            (dataframe['ADX'] > 25) &  # Add ADX filter for stronger trends
            (dataframe['ATR'] > dataframe['ATR'].rolling(window=14).mean()) &  # Filter for volatility
            (dataframe['long_vwap'] == 1.5) &
            (dataframe['STOCH_D'] < dataframe['STOCH_K']) &
            (dataframe['volume'] > 1000)
        ), ['enter_long', 'enter_tag']] = (1, 'rsi')
        
        
        # -------------------------- based on Con Di ---------------------------
        # dataframe.loc[(
        #     (dataframe['condi_wad_hma'] == -1) &
        #     (dataframe['condi_rsi_hma'] == -1) &
        #     (dataframe['condi_ema_hma'] == -1) &
        #     (dataframe['close'] < dataframe['HMA']) &
        #     (dataframe['ADX'] > 25) &  # تایید روند قوی با ADX
        #     (dataframe['STOCH_D'] > dataframe['STOCH_K']) &
        #     (dataframe['RSI'] > 70) &  # تایید حالت اشباع خرید برای Short
        #     (dataframe['volume'] > 1500) 
        #     # (dataframe['volume'] > dataframe['volume'].rolling(window=10).mean())  # بررسی حجم غیرعادی
        # ), ['enter_short', 'enter_tag']] = (1, 'condis')
        
        # dataframe.loc[(
        #     (dataframe['condi_wad_hma'] == 1) & 
        #     (dataframe['condi_rsi_hma'] == 1) &
        #     (dataframe['condi_ema_hma'] == 1) &
        #     (dataframe['close'] > dataframe['HMA']) &
        #     (dataframe['ADX'] > 25) &  # تایید روند قوی با ADX
        #     (dataframe['RSI'] < 30) &  # تایید حالت اشباع فروش برای Long
        #     (dataframe['STOCH_D'] < dataframe['STOCH_K']) &
        #     (dataframe['volume'] > 1500) 
        #     # (dataframe['volume'] > dataframe['volume'].rolling(window=10).mean())  # بررسی حجم غیرعادی
        # ), ['enter_long', 'enter_tag']] = (1, 'condis')
        
        
        # ---------------------------------kosh----------------------------------
        dataframe.loc[(
            (dataframe['p_mom'] == True) &
            (dataframe['p_mom'].shift(1) == False) &
            (dataframe['STOCH_D'] < dataframe['STOCH_K']) &
            (dataframe['WAD'] < -51)
        ), ['enter_long', 'enter_tag']] = (1, 'rmi')
        dataframe.loc[(
            (dataframe['n_mom'] == True) &
            (dataframe['n_mom'].shift(1) == False) &
            (dataframe['STOCH_D'] > dataframe['STOCH_K']) &
            (dataframe['WAD'] > -49)
        ), ['enter_short', 'enter_tag']] = (1, 'rmi')
        
        # ---------------------------------STOCH----------------------------------
        # dataframe.loc[(
        #     (dataframe['p_mom'] == True) &
        #     (dataframe['p_mom'].shift(1) == False) &
        #     (dataframe['WAD'] < -51)
        # ), ['enter_long', 'enter_tag']] = (1, 'rmi')
        # dataframe.loc[(
        #     (dataframe['n_mom'] == True) &
        #     (dataframe['n_mom'].shift(1) == False) &
        #     (dataframe['WAD'] > -49)
        # ), ['enter_short', 'enter_tag']] = (1, 'rmi')

        return dataframe

    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, 
                            time_in_force: str, entry_tag: str, **kwargs) -> bool:        
        # Limit the number of open trades with the same entry tag
        open_trades = self.get_open_trades()        
        open_trades_with_tag = [trade for trade in open_trades if trade.enter_tag == entry_tag]
        if len(open_trades_with_tag) >= self.entry_tags_limit:
            # logger.warning(f"Reached limit of {self.entry_tags_limit} trades with entry tag {entry_tag}.")
            return False
        return True

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(
            (dataframe['enter_short'] == 1 ) 
        ), ['exit_long', 'exit_tag']] = (1, 'Lopp_signal')
        dataframe.loc[(
            (dataframe['enter_long'] == 1 ) 
        ), ['exit_short', 'exit_tag']] = (1, 'Sopp_signal')
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[str]:

        dataframe, *_ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        current_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2]  # Get the previous row for comparison

        # if any(tag in trade.enter_tag for tag in ['lrsi', 'srsi']) and current_profit >= 0.02:
        #     return 'condidit'
        
        if any(tag in trade.enter_tag for tag in ['ketlner']) and (current_candle['close'] < current_candle['kc_lower']) and (current_candle['kama_slope'] > 1) and (previous_candle['kama_slope'] < 1):
            return "ketlner_did_it!!"
        
        if any(tag in trade.enter_tag for tag in ['ketlner']) and (current_candle['close'] > current_candle['kc_upper']) and (current_candle['kama_slope'] < 1) and (previous_candle['kama_slope'] > 1):
            return "ketlner_did_it!!"
        
        # if any(tag in trade.enter_tag for tag in ['Satf']) and (current_candle['kama_slope'] > previous_candle['kama_slope']):
        #     return 'satf ended'
        
        # if any(tag in trade.enter_tag for tag in ['Latf']) and (current_candle['kama_slope'] < previous_candle['kama_slope']):
        #     return 'latf ended'
        
        # if any(tag in trade.enter_tag for tag in ['S_vwap']) and (current_candle['typical_price'] > current_candle['vwap']):
        #     return 'svwap ended'
        
        # if any(tag in trade.enter_tag for tag in ['L_vwap']) and (current_candle['typical_price'] < current_candle['vwap']):
        #     return 'lvwap ended'
        
        # if any(tag in trade.enter_tag for tag in ['L_k*ema']) and (current_candle['kama_'] < current_candle['ema']) and (current_profit < 0):
        #     return "Lprevent!!"
        
        # if any(tag in trade.enter_tag for tag in ['S_k*ema']) and (current_candle['kama_'] > current_candle['ema']) and (current_profit < 0):
        #     return "Sprevent!!"

   
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                **kwargs) -> float:
        dataframe, *_ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        current_candle = dataframe.iloc[-1].squeeze()
        # dataframe['spread'] = 1000 * abs(dataframe['high'] - dataframe['low']) / dataframe['close']
        # # بقیه کدهای محاسبه leverage
        # dataframe['spread_avg'] = dataframe['spread'].rolling(window=100).mean()

        # leverage = self.max_leverage * (1.0 - float(current_candle['atr_pct']))
        return 15.0
        # return leverage

