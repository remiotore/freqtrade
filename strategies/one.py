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

class one(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    # Stoploss:
    stoploss = -0.05
    max_open_trades = 1     # Max Open Trades
    entry_tags_limit = 1   # limit per entry tag
    max_leverage = 5.0
    wave_length = 0
    # ROI table:
    minimal_roi = {
        "0": 0.10,     #     "0": 0.219,
        "10": 0.05,
        "20": 0.03,
        "31": 0.02,    #     "24": 0.087,
        "51": 0.01,    #     "67": 0.024,
        "60": 0         #     "164": 0
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
    pmom = 66 # 66
    nmom = 30 # 30
    

    
    #for calculation of coral trend
    ema_fast_period=50
    ema_slow_period=200
    # cci_period=IntParameter(30, 200, default=150, space= "buy", optimize=True)
    cci_threshold=IntParameter(0, 200, default=200, space = "buy", optimize=False)
    cci_period= 15
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.03 #---> 0.002 is Original Value
    trailing_stop_positive_offset = 0.15 #--> 0.055 is Original Value
    trailing_only_offset_is_reached = True
    
    # Sell signal
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.10
    ignore_roi_if_entry_signal = True
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe1 = '15m'
    informative_timeframe2 = '1h'
    informative_timeframe3 = '4h'
    informative_pair= 'BTC/USDT:USDT'
    process_only_new_candles = True
    startup_candle_count = 200
    plot_config = {
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
        # 15 min
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe1)
        informative['ema_20'] = ta.EMA(informative['close'], timeperiod=20)
        informative['rsi'] = ta.RSI(informative['close'], timeperiod=7)
        informative['sar'] = ta.SAR(informative['high'], informative['low'], acceleration=0.02, maximum=0.2)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe1, ffill=True)  
        # 1h
        informative2 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe2)  # داده‌های تایم‌فریم دوم
        informative2['nominal'] = informative2['close'] * informative2['volume']
        informative2['HMA'] = hma_volume(informative2, period= 5)
        informative2['macd'], informative2['macdsignal'], informative2 ['macdhist'] = ta.MACD(informative2['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe = merge_informative_pair(dataframe, informative2, self.timeframe, self.informative_timeframe2, ffill=True)
        # 4h
        informative3 = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe3)  # داده‌های تایم‌فریم دوم
        informative3['macd'], informative3['macdsignal'], informative3 ['macdhist']= ta.MACD(informative3['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe = merge_informative_pair(dataframe, informative3, self.timeframe, self.informative_timeframe3, ffill=True)
        
        # btc_price = self.dp.get_pair_dataframe(pair=self.informative_pair, timeframe=self.timeframe)
        # dataframe = merge_informative_pair(dataframe, btc_price, self.timeframe, self.timeframe, ffill=True)
        
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'volume']]
        drop_columns += [f"{s}_{self.informative_timeframe1}" for s in ['date', 'open', 'high', 'low', 'volume']]
        drop_columns += [f"{s}_{self.informative_timeframe2}" for s in ['date', 'open', 'high', 'low', 'volume']]
        drop_columns += [f"{s}_{self.informative_timeframe3}" for s in ['date', 'open', 'high', 'low', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)
        # ------------ 5m timeframe indicators ------------------------------
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=7)
        dataframe['macd'], dataframe['macdsignal'], _ = ta.MACD(dataframe['close'], fastperiod=6, slowperiod=13, signalperiod=5)
        dataframe['ema_9'] = ta.EMA(dataframe['close'], timeperiod=9)
        
        dataframe['nominal'] = dataframe['close'] * dataframe['volume']
        dataframe['HMA'] = hma_volume(dataframe, period= 5)
        dataframe['HMA12'] = ta.EMA(dataframe['HMA'], timeperiod=12)
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        vwap = (typical_price * dataframe['volume']).cumsum() / dataframe['volume'].cumsum()
        dataframe['vwap'] = vwap
        dataframe['ADX'] = ta.ADX(dataframe, timeperiod=21)
        dataframe['ATR'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['vwap_upper'] = dataframe['vwap'] + dataframe['ATR']
        dataframe['vwap_lower'] = dataframe['vwap'] - dataframe['ATR']
        dataframe['OBV'] =  ta.OBV(dataframe['close'], dataframe['volume'])
        
        dataframe['not_long'] = np.where((
            (dataframe['macdhist_4h'] < 0 ) &
            (dataframe['macdhist_4h'].shift(1) > dataframe['macdhist_4h']) &
            (dataframe['macdhist_1h'] < 0 ) &
            (dataframe['macdhist_1h'].shift(1) > dataframe['macdhist_1h']) 
            ),0 , 1)
        dataframe['not_short'] = np.where((
            (dataframe['macdhist_4h'] > 0 ) &
            (dataframe['macdhist_4h'].shift(1) < dataframe['macdhist_4h']) &
            (dataframe['macdhist_1h'] > 0 ) &
            (dataframe['macdhist_1h'].shift(1) < dataframe['macdhist_1h']) 
            ),0 , 1)
      
        return dataframe
    
    def get_open_trades(self):
        # Fetch all open trades
        return Trade.get_open_trades()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # -------------------------- based on Con Di ---------------------------
        dataframe.loc[(
            # (dataframe['close'] > dataframe['vwap_upper']) &  
            (dataframe['HMA12'] > dataframe['HMA_1h']) &  
            (dataframe['HMA12'].shift(1) < dataframe['HMA_1h'].shift(1)) &  
            (dataframe['ADX'] > 30) & 
            (dataframe['not_long'] == 0) &
            (dataframe['OBV'] > dataframe['OBV'].shift(1)) 
        ), ['enter_long', 'enter_tag']] = (1, 'long')
        
        dataframe.loc[(
            # (dataframe['close'] < dataframe['vwap_lower']) &  
            (dataframe['HMA12'] < dataframe['HMA_1h']) &  
            (dataframe['HMA12'].shift(1) > dataframe['HMA_1h'].shift(1)) &  
            (dataframe['ADX'] > 30) & 
            (dataframe['not_short'] == 0) &
            (dataframe['OBV'] < dataframe['OBV'].shift(1)) 
        ), ['enter_short', 'enter_tag']] = (1, 'short')
 
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
        return dataframe
    
    # def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[str]:

    #     dataframe, *_ = self.dp.get_analyzed_dataframe(
    #         pair=pair, timeframe=self.timeframe
    #     )
    #     current_candle = dataframe.iloc[-1].squeeze()
    #     previous_candle = dataframe.iloc[-2]  # Get the previous row for comparison

   
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                **kwargs) -> float:
        dataframe, *_ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        current_candle = dataframe.iloc[-1].squeeze()


        leverage = self.max_leverage * (1 - float(current_candle['ATR']/100))
        return leverage
        # return leverage