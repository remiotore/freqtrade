# --- Do not remove these libs ---
from typing import List
from skopt.space import Dimension, Integer
from datetime import datetime, timedelta, timezone
import numpy as np  # noqa
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, merge_informative_pair, BooleanParameter
from functools import reduce
from freqtrade.persistence import Trade
import pandas as pd  # noqa
#from technical.indicators import zema, ema, vwma, chaikin_money_flow, VIDYA, PMAX
from technical.indicators import zema, ema, vwma, chaikin_money_flow, VIDYA
import math



# --------------------------------

"""
DO NOT USE TO DO REAL TRADES - THIS IS A FIRST ATTEMPT AT WRITING A STRATEGY FOR FREQTRADE
"""
class RSIDip(IStrategy):


    # Minimal ROI designed for the strategy.
    # adjust based on market conditions. We would recommend to keep it low for quick turn arounds
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 100.0
    }


    # Optimal stoploss designed for the strategy  
    stoploss = -1.00
        
    #trailing_stop = True
    #trailing_stop_positive = 0.02
    # Trailing stop:


    # Optimal ticker interval for the strategy
    ticker_interval = '1h'


    # buy parameters
    ############################################################################################################################
    
    # ema lines
    buy_ema50_period = IntParameter(10, 150, default=50, space='buy', optimize=False, load=True)
    buy_ema200_period = IntParameter(150, 300, default=200, space='buy', optimize=False, load=True)

    
    # price monitoring window
    buy_short_window = IntParameter(1, 300, default=72, space='buy', optimize=False, load=True)
    buy_short_perc = IntParameter(1, 25, default=25, space='buy', optimize=False, load=True)
    
    
    buy_rsi_sma_period = IntParameter(3, 8, default=4, space='buy', optimize=True, load=True)
    buy_rsi_time_period = IntParameter(3, 25, default=14, space='buy', optimize=False, load=True)
    buy_std_dev_nbdev = DecimalParameter(1.5, 2, default=1, space='buy', decimals=1, optimize=True, load=True)

    buy_trend_limit = DecimalParameter(0.000, 0.003, default=0.00000, space='buy', decimals=3, optimize=False, load=True)			    
    buy_trend_long_limit = DecimalParameter(0.000, 0.003, default=0.000, space='buy', decimals=5, optimize=False, load=True)			        
        
    buy_count_limit = IntParameter(1, 5, default=3, space='buy', optimize=True, load=True)			
    buy_candle_count_limit = IntParameter(1, 100, default=100, space='buy', optimize=True, load=True)			
    
    # sell parameters
    ##############################################################################################################################
    sell_atr_stoploss = DecimalParameter(0.5, 2.0, default=1.5, space='sell', decimals=1, optimize=True, load=True)			
    sell_atr_target = DecimalParameter(0.5, 3.0, default=2.7, space='sell', decimals=1, optimize=True, load=True)			
    sell_reduced_atr_target_alpha_percentage = DecimalParameter(0.5, 0.90, default=0.9, space='sell', decimals=1, optimize=True, load=True)
    sell_reduced_atr_target_beta_percentage = DecimalParameter(0.1, 0.50, default=0.3, space='sell', decimals=1, optimize=True, load=True)
    sell_trade_candle_count_limit = IntParameter(1, 100, default=20, space='sell', optimize=True, load=True)
    sell_latest_concurrent_loss_count_limit = IntParameter(1, 100, space='sell', default=2, optimize=True, load=True)


    ###############
    # custom_data
    custom_data = {}

    class PairTracker:

        trade = None
        pair = ""
        highest_price = 0
        highest_profit = 0
        latest_price = 0
        latest_profit = 0
        wins = 0
        loses = 0
        last_was = "unknown"

        highest_stoploss_rate = -9999999
        reduced_atr_target_alpha_breached = False

        latest_concurrent_loss_count = 0

        DEBUG = False
        debug_pair = 'ETC/USDT'
        
        def __init__(self, pair):
            self.pair = pair


        def __str__ (self):                    
            t = ""
            return 'Custom_Pair(pair=' + str(self.pair)+ ' ,highest_price=' + str(self.highest_price)+ ' ,highest_profit=' \
            + str(self.highest_profit) \
            + ', latest_price=' + str(self.latest_price) \
            + ', latest_profit=' + str(self.latest_profit) \
            + ', wins=' + str(self.wins) + ', loses=' + str(self.loses) \
            + ', last_was=' + str(self.last_was) \
            + ', highest_stoploss_rate=' + str(self.highest_stoploss_rate) \
            + ', reduced_atr_target_alpha_breached=' + str(self.reduced_atr_target_alpha_breached) \
            + ', latest_concurrent_loss_count=' + str(self.latest_concurrent_loss_count) \
            + ' )'
                 
            
        def process_latest_price(self, price):
            self.latest_price = price
            if price > self.highest_price or price is None:
                self.highest_price = price
             
        def process_latest_profit(self, profit):
            self.latest_profit = profit
            if profit > self.highest_profit or profit is None:
                self.highest_profit= profit
                
        
        def trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, current_time: datetime):
            
            debug = (pair==self.debug_pair) and self.DEBUG
            
            if(debug):
                print("trade_entry "+" pair="+str(pair)+", order_type="+str(order_type)+", amount="+str(amount)+", rate="+str(rate)+", time_in_force="+
                    str(time_in_force)+", current_time="+str(current_time))
            pass
            
        def trade_exit(self, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime):
                                           
                debug = (trade.pair==self.debug_pair) and self.DEBUG
                
                if debug:                
                    print ("---------------------------------------------------------------")
                    print("trade_exit called "+"pair="+str(trade.pair)+", trade="+str(trade)+
                        ", order_type="+str(order_type)+", amount="+str(amount)+
                        ", rate="+str(rate)+", time_in_force="+str(time_in_force)+
                        ", sell_reason="+str(sell_reason)+", current_time="+str(current_time))                           
                    print ("\n")
                    print(self)
                    #print("sell reason :"+self.trade.sell_reason)
                    print("***********************************\n\n")        
                
                if self.latest_profit > 0:
                    if debug:                
                        print("**WIN")
                    self.wins += 1
                    self.last_was = "win"
                    self.latest_concurrent_loss_count = 0
                else:
                    if debug:                
                        print("**LOSS")
                    self.loses +=1
                    self.last_was = "loss"
                    self.latest_concurrent_loss_count += 1
                
                # reset
                self.highest_stoploss_rate = -9999999  
                self.reduced_atr_target_alpha_breached = False  
                #self.trade_candle_count = 0
                    
        
        def process_trade(self, trade, price, profit):
        
            debug = (trade.pair==self.debug_pair) and self.DEBUG
                    
            if self.trade is None:
                self.trade = trade
        
            if self.trade.open_date != trade.open_date:
            
                if debug:
                    print("process_trade called ****************************************")
                    print(self)
                    #print("sell reason :"+self.trade.sell_reason)
                    print("*********************************\n")
                """
                if self.latest_profit > 0:
                    self.wins += 1
                    self.last_was = "win"
                else:
                    self.loses +=1
                    self.last_was = "loss"
                """ 
                self.trade = trade
                self.price = price
                self.profit = profit
                
                #self.trade_candle_count = 1
                
            else:
                #self.trade_candle_count += 1
                self.process_latest_price(price)
                self.process_latest_profit(profit)
                

        def latest_stoploss_rate(self, stoploss_rate):
            if self.highest_stoploss_rate < stoploss_rate:
                debug = (self.pair==self.debug_pair) and self.DEBUG
                if(debug):
                    print("old stoploss rate\t"+str(self.highest_stoploss_rate)+"\tnew stoploss rate\t"+str(stoploss_rate))
                self.highest_stoploss_rate = stoploss_rate  

        
        

        def reduced_atr_target_check(self, current_rate, bought, candle_count_limit, concurrent_loss_limit):
            

            if self.reduced_atr_target_alpha_breached == True:
                if (current_rate >= bought['reduced_atr_target_alpha']).bool():
                    return 'reduced_atr_target_alpha_rentry'

            
            #if self.trade_candle_count >= candle_count_limit and (current_rate > bought['reduced_atr_target_beta']).bool():
            if (bought['crossedOverGroupCandleCount'] >= candle_count_limit).bool() and (current_rate > bought['reduced_atr_target_beta']).bool():            
                return 'reduced_atr_target_beta_due_candle_count_limit'
            
            
            if self.latest_concurrent_loss_count >= concurrent_loss_limit and (current_rate > bought['reduced_atr_target_beta']).bool():
                return 'reduced_atr_target_beta_due_to_concurrent_loss_limit'
            
            
            """
            if self.reduced_atr_target_alpha_breached == True and self.trade_candle_count >= candle_count_limit:
                if (current_rate >= bought['reduced_atr_target_alpha']).bool():
                    return 'reduced_atr_target_rentry'
            """
            
            if self.reduced_atr_target_alpha_breached == False:
                if (current_rate > bought['reduced_atr_target_alpha']).bool():
                    self.reduced_atr_target_alpha_breached = True
            
                
            return None

           
    ############### PairTracker END ********************************************************
    
    
 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        # add to custom data--------------------------
        if not metadata["pair"] in self.custom_data:
            # Create empty entry for this pair
            self.custom_data[metadata["pair"]] = {'tracking_data':RSIDip.PairTracker(metadata["pair"])}
        #=============================================
        
        self.pop_indicators(dataframe)
        
        return dataframe


    """
    populating indicators in seperate method as want to call when performing optimisation from the 
    populate_buy_trend method etc...
    """
    def pop_indicators(self, dataframe: DataFrame):
        
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=int(self.buy_ema50_period.value))
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=int(self.buy_ema200_period.value))


        dataframe['trend'] = dataframe['close'].rolling(window=72, min_periods=self.buy_short_window.value, center=False).apply(lambda x: generateTrend(x))        
        dataframe['trendLong'] = dataframe['close'].rolling(window=144, min_periods=144, center=False).apply(lambda x: generateTrend(x))

        
        rsi_time_period = int(self.buy_rsi_time_period.value)
        
        # was 14
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=int(self.buy_rsi_time_period.value))


        dataframe['stoploss_for_candle'] = dataframe[['open','close']].min(axis=1) - (dataframe['atr']*self.sell_atr_stoploss.value)
        dataframe['atr_target'] = dataframe[['open','close']].min(axis=1) + (dataframe['atr']*self.sell_atr_target.value)
        dataframe['reduced_atr_target_alpha'] = dataframe[['open','close']].min(axis=1) + (dataframe['atr']*(self.sell_atr_target.value*self.sell_reduced_atr_target_alpha_percentage.value))
        dataframe['reduced_atr_target_beta'] = dataframe[['open','close']].min(axis=1) + (dataframe['atr']*(self.sell_atr_target.value*self.sell_reduced_atr_target_beta_percentage.value))


        dataframe["short_rolling_high_max"] = dataframe["close"].rolling(self.buy_short_window.value).max()        
        dataframe["short_rolling_low_min"] = dataframe["open"].rolling(self.buy_short_window.value).min()
        dataframe["short_high_low_middle_point"] = ((dataframe["short_rolling_high_max"] - dataframe["short_rolling_low_min"])/2) + dataframe["short_rolling_low_min"]
        
        dataframe["close_mean"] = dataframe["close"].rolling(self.buy_short_window.value).mean()        

        dataframe['std_dev_mean'] = ta.STDDEV(dataframe['close_mean'], timeperiod=self.buy_short_window.value, nbdev=2.0)
        dataframe['buy_below'] = dataframe["close_mean"] - dataframe['std_dev_mean']
        
        dataframe['ema_short_window'] = ta.EMA(dataframe['close'], timeperiod=self.buy_short_window.value)
        dataframe['ema_std_dev_mean'] = ta.STDDEV(dataframe['ema_short_window'], timeperiod=self.buy_short_window.value, nbdev=1.0)
        dataframe['ema_buy_below'] = dataframe["ema_short_window"] - dataframe['ema_std_dev_mean']
        
        
        dataframe["short_high_low_buy_below"] = (((dataframe["short_rolling_high_max"] - dataframe["short_rolling_low_min"])/100)*self.buy_short_perc.value) + dataframe["short_rolling_low_min"]

        #RSI 14
        #moved higher up rsi_time_period = int(self.buy_rsi_time_period.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=rsi_time_period)


        dataframe['rsiDiff'] = dataframe['rsi'].shift(1) - dataframe['rsi']
        
        avgperiod = int(self.buy_rsi_sma_period.value)
        dataframe['rsisma'] = ta.SMA(dataframe['rsiDiff'], timeperiod=avgperiod)        
        dataframe['stddev'] = ta.STDDEV(dataframe['rsiDiff'], timeperiod=avgperiod, nbdev=float(self.buy_std_dev_nbdev.value))
        dataframe['plusstddev'] = dataframe['rsisma'] + dataframe['stddev']
        

        # default for ADX according to investopedia is 14
        #
        # ADX Value	Trend Strength
        # 0-25	Absent or Weak Trend
        # 25-50	Strong Trend
        # 50-75	Very Strong Trend
        # 75-100	Extremely Strong Trend
        #
        adx_time_period = int(self.buy_rsi_time_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=adx_time_period)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=adx_time_period)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=adx_time_period)
         
         
       
    def add_custom_columns(self, dataframe: DataFrame): 

        # this needs to be the same as whats in populate_buy_trend
        # need it here as dependant on buy
        dataframe.loc[
            (
                    
                    (dataframe['ema50'] > dataframe['ema200']) & 
                    (dataframe['rsiDiff'] > dataframe['plusstddev']) &                     
                    (dataframe['volume'] > 0)

            ),
            'buyTEST'] = 1    
    
    
        # set row number column
        dataframe['rownum'] = np.arange(len(dataframe))

        # identify crossovergroup starts
        dataframe.loc[(dataframe['ema50'] > dataframe['ema200'])&(dataframe['ema50'].shift(1) <= dataframe['ema200'].shift(1)), 'crossedOverGroup'] = dataframe['rownum']
            
        # fill forward so rows have crossedOverGroup they belong to
        dataframe['crossedOverGroup'] = dataframe['crossedOverGroup'].fillna(method="ffill")
         
        # add candle count for crossed over group
        dataframe['crossedOverGroupCandleCount'] = dataframe['rownum'] - dataframe['crossedOverGroup']
        
        # add crossed over count of buys
        dataframe["crossedOverGroupBuyCount"] = group = dataframe['buyTEST'].groupby(dataframe['crossedOverGroup']).cumsum()
       

        

    #def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:   
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:    
        
        self.pop_indicators(dataframe)

        """
        pair = metadata['pair']
        pair_data = self.custom_data.get(pair)
        tracked_pair = pair_data.get('tracking_data')
                
        last_candle = dataframe.iloc[-1].squeeze()
        candle_before = dataframe.iloc[-2].squeeze()
        """
        
        # custon columns needed for making buy decision so populate'em
        self.add_custom_columns(dataframe)        
    
        dataframe.loc[
            (
                    
                    (dataframe['ema50'] > dataframe['ema200']) & 
                    (dataframe['rsiDiff'] > dataframe['plusstddev']) &                     
                    (dataframe['crossedOverGroupCandleCount'] < self.buy_candle_count_limit.value) &
                    (dataframe['crossedOverGroupBuyCount'] < self.buy_count_limit.value) &                    
                    (dataframe['volume'] > 0)

            ),
            'buy'] = 1

        """
        uncomment lines for debugging if needed
        """
        #self.add_custom_columns(dataframe)                    
        #dataframe.to_csv('/freqtrade/user_data/temp/dataframe-{}.csv'.format(metadata['pair'].replace("/","-")))            
        
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
    
        self.pop_indicators(dataframe)
    
        # Never sells
        #dataframe.loc[:, 'sell'] = 0

        # sell signal to get out if the ema's have crossed back
        dataframe.loc[
            (
                    (dataframe['ema50'] <= dataframe['ema200']) & 
                    (dataframe['ema50'].shift(1) > dataframe['ema200'].shift(1)) & 
                    (dataframe['volume'] > 0)
            ),
            'sell'] = 1

        """
        uncomment lines for debugging if needed
        """
        #self.add_custom_columns(dataframe)                    
        #dataframe.to_csv('/freqtrade/user_data/temp/dataframe-{}.csv'.format(metadata['pair'].replace("/","-")))            
        
        return dataframe

     
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs):   

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)            
        
        # candles interested in        
        bought = dataframe[dataframe['date']==trade.open_date]            
        last_candle = dataframe.iloc[-1].squeeze()
        candle_before = dataframe.iloc[-2].squeeze()

        pair_data = self.custom_data.get(pair)
        tracked_pair = pair_data.get('tracking_data')
                
        tracked_pair.process_trade(trade, current_rate, current_profit)
        tracked_pair.latest_stoploss_rate(last_candle['stoploss_for_candle'])


        if tracked_pair.reduced_atr_target_check(current_rate, bought, self.sell_trade_candle_count_limit.value, self.sell_latest_concurrent_loss_count_limit.value) is not None:
            return tracked_pair.reduced_atr_target_check(current_rate, bought, self.sell_trade_candle_count_limit.value, self.sell_latest_concurrent_loss_count_limit.value)
        
        if (current_rate <= tracked_pair.highest_stoploss_rate):
            return 'sell_trailing_stoploss_rate'
        
        if ((current_rate <= bought['stoploss_for_candle'])).bool():        
            return 'sell_stoploss_on_bought_candle'
        
        if (current_rate >= bought['atr_target']).bool():
            return 'sell_atr_target'
            
        return None

    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        
        trade_monitor_pair = self.custom_data.get(pair)
        trade_monitor = trade_monitor_pair.get('tracking_data')
                                              
        trade_monitor.trade_entry(pair, order_type, amount, rate, time_in_force, current_time)        
        return True
        
        

    def confirm_trade_exit(self, pair:str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        trade_monitor_pair = self.custom_data.get(pair)
        trade_monitor = trade_monitor_pair.get('tracking_data')
        
        trade_monitor.trade_exit(trade, order_type, amount,
                           rate, time_in_force, sell_reason,
                           current_time)                

        return True



def generateTrend(data):

    if sum(~np.isnan(x) for x in data) < 2:
        return np.NaN

    df = pd.DataFrame(data)
    coefficients, residuals, _, result, _ = np.polyfit(range(len(data)),df[0],1,full=True)          
    return coefficients[-2]


