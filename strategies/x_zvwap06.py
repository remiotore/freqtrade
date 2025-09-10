import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta
import numpy as np
from functools import reduce
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, RealParameter,IntParameter,informative
from pandas import DataFrame, Series
from datetime import datetime
import math
import logging
from freqtrade.persistence import Trade
import pandas_ta as pta
from technical.indicators import RMI
import technical.indicators as ftt
import threading
import requests
from freqtrade.vendor.qtpylib import indicators


logger = logging.getLogger(__name__)
log = logging.getLogger(__name__)


def top_percent_change_dca(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']










def calc_zvwap(dataframe, pds, source1):
    volume = dataframe['volume'].rolling(pds).mean()
    close = dataframe['close'].rolling(pds).mean()
    
    mean = indicators.sma(volume * source1, pds) / indicators.sma(volume, pds)
    vwapsd = indicators.sma(pow(source1 - mean, 2), pds).apply(lambda x: pow(x, 0.5))
    zvwap = (close - mean) / vwapsd
    
    return zvwap

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')




class x_zvwap06(IStrategy):
    """
    PASTE OUTPUT FROM HYPEROPT HERE
    Can be overridden for specific sub-strategies (stake currencies) at the bottom.
    """

    buy_params = {






















        "tcp_percent_4_value": 0.04,
        "cti": -0.087,
        "buy_rsi": 35
      

      
      
      

    }

    sell_params = {

      "pHSL": -0.397,
      "pPF_1": 0.012,
      "pPF_2": 0.07,
      "pSL_1": 0.015,
      "pSL_2": 0.068,
      "sell_bbmiddle_close": 1.0909210168690215,
      "sell_fisher": 0.46405736994786184,
      
      "base_nb_candles_sell": 22,
      "high_offset": 1.014,
      "high_offset_2": 1.01,

      "sell_u_e_2_cmf": -0.0,
      "sell_u_e_2_ema_close_delta": 0.016,
      "sell_u_e_2_rsi": 10,






      "sell_deadfish_bb_factor": 1.002,
      "sell_deadfish_bb_width": 0.031,
      "sell_deadfish_profit": -0.109,
      "sell_deadfish_volume_factor": 1.512
      
      
    }

    minimal_roi = {
        "0": 100
    }
    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            }


        ]

    stoploss = -0.39  # use custom stoploss

    trailing_stop = True
    trailing_stop_positive = 0.006 #povodne 0.001
    trailing_stop_positive_offset = 0.08 #povodne 0.012
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    custom_info = {}

    use_custom_stoploss = False

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergencyexit': 'market',
        'forceentry': "market",
        'forceexit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    
    tcp_percent_4_value = DecimalParameter(0.01, 0.19, default=0.04 , space='buy', optimize = False)
    cti = DecimalParameter(-0.9, -0.0, default=-0.6 , optimize = False)
    buy_rsi = IntParameter(15, 30, default=35, optimize = False)

    dump= DecimalParameter(-1, 0, default=-0.15, space='buy', optimize=True)
    dump_buy= DecimalParameter(0, 0.5, default=0.15, space='buy', optimize=True)

    range_filter_length = IntParameter(10, 200, default=200, space='buy', optimize= False)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=True)
    delay_time = IntParameter(90, 1440, default=300, space='sell', optimize=True)
    fask_trailing = DecimalParameter(0.001, 0.02, default=0.001, space='sell', optimize=True)

    sell_fisher = RealParameter(0.1, 0.5, default=0.38414, space='sell', optimize=False)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=1.07634, space='sell', optimize=False)

    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.08 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , space='sell', optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.5 ,space='sell', optimize = is_optimize_deadfish)


    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)

    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)
    
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.11, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.015, space='sell', decimals=3, optimize=False, load=True)

    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', optimize=False, load=True)

    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=False, load=True)

    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell',optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=False,load=True)
    
    antipump_threshold = DecimalParameter(0, 0.4, default=0.15, space='buy', optimize=False)



    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:





        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)
        dataframe['smi'], dataframe['smi_ma'], dataframe['smi_trend'] = smi_trend(dataframe, 9, 3, 'EMA', 10)


        return dataframe
    
    def colored(self,param, color):
        color_codes = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
        }
        return color_codes[color] + param + '\033[0m'
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        filled_entries = trade.select_filled_orders(trade.entry_side)
        min_profit_rate = trade.max_rate if trade.is_short else trade.min_rate
        max_profit_rate = trade.min_rate if trade.is_short else trade.max_rate

        min_profit = (trade.calc_profit_ratio(min_profit_rate) * 100)
        max_profit = (trade.calc_profit_ratio(max_profit_rate) * 100)

        profit = (trade.calc_profit_ratio(rate) * 100)
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        ll = self.custom_info[f'{pair}']['Level']
        
        if profit < 0:
            if ll == '0' or ll == 0 :
                log.info(
                f'EXIT Pair: {pair:13} | Level: {count_of_buys} | Exit Reason: {exit_reason} | Min_profit: {min_profit:.2f}% | Max_profit: {max_profit:.2f}% |  profit: {profit:.2f}% | {current_time}')
            else:
                self.log(
                f'EXIT Pair: {pair:13} | Level: {count_of_buys} | Exit Reason: {exit_reason} | Min_profit: {min_profit:.2f}% | Max_profit: {max_profit:.2f}% |  profit: {profit:.2f}% | {current_time}',color='red')
        else:
            self.log(
                f'EXIT Pair: {pair:13} | Level: {count_of_buys} | Exit Reason: {exit_reason} | Min_profit: {min_profit:.2f}% | Max_profit: {max_profit:.2f}% |  profit: {profit:.2f}% | {current_time}',color='green')
            self.custom_info[f'{pair}']['Level'] = 0

        
        if self.config['telegram']['enabled'] == True:
            min_profit_rate = trade.max_rate if trade.is_short else trade.min_rate
            max_profit_rate = trade.min_rate if trade.is_short else trade.max_rate

            min_profit = (trade.calc_profit_ratio(min_profit_rate) * 100)
            max_profit = (trade.calc_profit_ratio(max_profit_rate) * 100)

            start_time = datetime.now()
            self.dp.send_msg(
                f"↕️ {pair} Min profit:  {min_profit:.2f}%  Max profit:  {max_profit:.2f}%\n"
                f"💰 {pair} Open fee:  {trade.fee_open * 100:.4f}%  Close fee:  {trade.fee_close * 100:.4f}%"
            )
            logger.info(f"{pair} took {datetime.now() - start_time} to send telegram message")

        return True
        

    def log(self, param, color):
        log.info(self.colored(param, color))








    
    
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        buy_tag = ''
        if hasattr(trade, 'enter_tag') and trade.enter_tag is not None:
            buy_tag = trade.enter_tag








        if (last_candle is not None):













            if (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_1.value)):
                return 'trail_target_1'
            elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_2.value)):
                return 'trail_target_2'
            elif (current_profit > 3) & (last_candle['rsi'] > 85):
                 return 'RSI-85 target'




        


   
            
            if (current_profit > 0.05) & (count_of_buys < 4)   & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) & (last_candle['rsi']>50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return f"sell signal1( {buy_tag})"
            if (current_profit > 0.05) & (count_of_buys >= 4)  & (last_candle['close'] > last_candle['hma_50'] * 1.01) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) & (last_candle['rsi']>50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return f"sell signal1 * 1.01( {buy_tag})"
            if (current_profit > 0.05) & (count_of_buys < 4)  & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)):
                return f"sell signal2( {buy_tag})"
            if (current_profit > 0.05) & (count_of_buys >= 4)  & (last_candle['close'] > last_candle['hma_50'] * 1.01) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &  (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']) :
                return f"sell signal2 * 1.01( {buy_tag})"




            if (current_profit > 0.05)  & (last_candle['rsi'] < 42.0) &  (last_candle['cmf'] < -0.4):
                return f"profit exit1( {buy_tag})"
            elif (current_profit > 0.05)  & (last_candle['rsi'] < 43.0) &  (last_candle['cmf'] < -0.0) &  (last_candle['cmf_15m'] < -0.0) &  (last_candle['cmf_1h'] < -0.0) :
                return f"profit exit2( {buy_tag})"
            elif (current_profit > 0.05)  & (last_candle['rsi'] < 41.0) &  (last_candle['cmf'] < -0.2) &  (last_candle['cmf_1h'] < -0.0):
                return f"profit exit3( {buy_tag})"
            elif (current_profit > 0.05)  & (last_candle['rsi'] < 44.0) &  (last_candle['cmf'] < -0.1) & (last_candle['cmf_15m'] < -0.1) &  (last_candle['cmf_1h'] < -0.1)   :
               return f"profit exit4( {buy_tag})"
            elif (current_profit > 0.05)  & (last_candle['rsi'] < 40.0) &  (last_candle['cmf'] < -0.2) & (last_candle['cmf_15m'] < -0.2):
                return f"pprofit exit5( {buy_tag})"
            elif (current_profit > 0.05)  & (last_candle['rsi'] < 43.0) &  (last_candle['cmf'] < -0.4) &  (last_candle['cmf_15m'] < -0.0) & (last_candle['cmf_1h'] < -0.0) :
                return f"profit exit6( {buy_tag})"

            elif (current_profit > 0.02)  & (last_candle['rsi'] < 30.0) &  (last_candle['cmf'] < -0.4):
                return f"profit exit bear1( {buy_tag})"
            elif (current_profit > 0.03)  & (last_candle['rsi'] < 35.0) &  (last_candle['cmf'] < -0.4):
                return f"profit exit bear2( {buy_tag})" 
            elif (current_profit > 0.01)  & (last_candle['rsi'] < 25.0) &  (last_candle['cmf'] < -0.6):
                return f"profit exit bear3( {buy_tag})"

            if (    (current_profit < self.sell_deadfish_profit.value)

                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (last_candle['cmf'] < 0.0)

            ):
                return f"sell_stoploss_deadfishD( {buy_tag})"            








            


    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        if current_time - timedelta(minutes=int(self.delay_time.value)) > trade.open_date_utc:
            if current_profit >= -0.01:
                return -0.003

        if current_time - timedelta(minutes=int(self.delay_time.value) * 2) > trade.open_date_utc:
            if current_profit >= -0.02:
                return -0.006
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return self.fask_trailing.value



        return self.stoploss
    
    def is_support(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data)-1):
            if row < len(row_data)/2:
                conditions.append(row_data[row] > row_data[row+1])
            else:
                conditions.append(row_data[row] < row_data[row+1])
        return reduce(lambda x, y: x & y, conditions)



    @informative('5m', 'BTC/{stake}')
    def populate_indicators_btc_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['btc_close'] = dataframe['close']
        dataframe['btc_rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['btc_ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['btc_ema_slow'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['down'] = (dataframe['btc_ema_fast'] < dataframe['btc_ema_slow']).astype('int')
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:




        
        dataframe['smi'], dataframe['smi_ma'], dataframe['smi_trend'] = smi_trend(dataframe, 9, 3, 'EMA', 10)

        dataframe[f'range{self.range_filter_length.value}'] = range_filter(dataframe=dataframe,length=self.range_filter_length.value)

        if not metadata["pair"] in self.custom_info:

            self.custom_info[metadata["pair"]] = {}
            self.custom_info[metadata["pair"]]['Level'] = {}


        dataframe  ['test'] =  (
                (dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.125 )) &
                ( (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.225 )) )
            )








        for val in self.base_nb_candles_sell.range:
             dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])






        dataframe['zvwap'] = calc_zvwap(dataframe, pds=14, source1=dataframe['close'])

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_fast1'] = dataframe['rsi_fast'].shift(1)
        dataframe['rsi_fast2'] = dataframe['rsi_fast'].shift(2)
        dataframe['rsi_fast_avg'] = (dataframe['rsi_fast'] + dataframe['rsi_fast1'] + dataframe['rsi_fast2']) / 3

        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']





























        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100

        dataframe['ema_vwap_diff_50'] = ( ( dataframe['ema_50'] - dataframe['vwap_lowerband'] ) / dataframe['ema_50'] )

        dataframe['tpct_change_0']   = top_percent_change_dca(dataframe,0)
        dataframe['tpct_change_1']   = top_percent_change_dca(dataframe,1)
        dataframe['tcp_percent_4'] =   top_percent_change_dca(dataframe , 4)

        dataframe['tpct_change_0']   = top_percent_change_dca(dataframe, 0)
        dataframe['tpct_change_1']   = top_percent_change_dca(dataframe, 1)
        dataframe['tpct_change_2']   = top_percent_change_dca(dataframe, 2)
        dataframe['tpct_change_4']   = top_percent_change_dca(dataframe, 4)

       
        dataframe['tpct_change_12']   = top_percent_change_dca(dataframe, 12)
        dataframe['tpct_change_144']   = top_percent_change_dca(dataframe, 144)







        dataframe['cti'] = pta.cti(dataframe["close"], length=20)











        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)
        
        dataframe['zvwap']= calc_zvwap(dataframe, pds=14, source1=dataframe['close'])
        dataframe['zvwap_dip'] = (dataframe['zvwap'] <= -0.5)

        dataframe['zvwap_cross'] = (dataframe['zvwap'] > 0) & (dataframe['zvwap'].shift(1) < 0)
        dataframe['zvwap_buy'] = dataframe['zvwap_dip']  &  dataframe['zvwap_cross']


        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)



        informative['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        informative['cmf'] = chaikin_money_flow(dataframe, 20)


        informative['roc'] = ta.ROC(informative, timeperiod=9)
        informative['live_data_ok'] = (dataframe['volume'].rolling(window=48, min_periods=48).min() > 0)
        informative['tcp_percent_72'] = self.top_percent_change_dca(dataframe,72)







        informative['cti_40'] = pta.cti(informative["close"], length=40)
        
        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['hma_5'] = qtpylib.hull_moving_average(dataframe['close'], window=30)
        dataframe['hma_4'] = qtpylib.hull_moving_average(dataframe['close'], window=15)

        dataframe['long_term_price_drop'] = np.where(
            (

                (
                    dataframe['close'].rolling(4).max() * 0.96 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                 (
                    dataframe['close'].rolling(6).max() * 0.96 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                 (
                    dataframe['close'].rolling(6).max() * 0.95 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                 (
                    dataframe['close'].rolling(8).max() * 0.95 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                (
                    dataframe['close'].rolling(8).max() * 0.94 > dataframe['close'].rolling(6).mean()
                ) |

                (
                    dataframe['close'].rolling(12).max() * 0.92 > dataframe['close'].rolling(6).mean()
                ) |

                (
                    dataframe['close'].rolling(24).max() * 0.90 > dataframe['close'].rolling(6).mean()
                ) 
            ), 1, 0)
        
        dataframe['nfixdip'] = np.where(
            (dataframe['tpct_change_0'] < 0.032) |    
            (dataframe['tpct_change_2'] < 0.06) |
            (dataframe['tpct_change_12'] < 0.24) |
            (dataframe['tpct_change_144'] < 0.6),
            1,
            0
        )
        
        dataframe['deadfish'] = np.where(
            (dataframe['close'] < dataframe['ema_200']) &
            (dataframe['bb_width'] < self.sell_deadfish_bb_width.value) &
            (dataframe['close'] > dataframe['bb_middleband2'] * self.sell_deadfish_bb_factor.value) &
            (dataframe['cmf'] < 0.0) &
            (dataframe['ha_open'] > dataframe['ha_close']),
            1, 0)  
        
        dataframe["deadfish2"] = dataframe["deadfish"].shift(1)
        dataframe["deadfish3"] = dataframe["deadfish"].shift(2)  
        dataframe["deadfish4"] = dataframe["deadfish"].shift(3)
        
        dataframe['pump_protection_strict'] = np.where(
                (dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.125 )) &
                ( (dataframe['close'].rolling(288).max() >= (dataframe['close'] * 1.225 )) ),
            1,
            0
        )              

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''



        """
        Generates a buy signal based on the BUY Rule conditions.
        """




        fast_ema = ta.EMA(dataframe['close'], window=13)
        slow_ema = ta.EMA(dataframe['close'], window=55)
        zvwap = calc_zvwap(dataframe, pds=14, source1=dataframe['close'])

        dip_condition = zvwap <= -0.5
        cross_above_zero_condition = (zvwap > 0) & (zvwap.shift() < 0)
        ema_condition = fast_ema > slow_ema
        
        

        dip_nfix = (




            (dataframe['tpct_change_0'] < 0.032) &    
            (dataframe['tpct_change_2'] < 0.06) &
            (dataframe['tpct_change_12'] < 0.24) &
            (dataframe['tpct_change_144'] < 0.6)
            
        )



        

        
        
        dataframe.loc[
                (


                (dataframe['close'] < zvwap) &

                (dataframe['tcp_percent_4'] > self.tcp_percent_4_value.value )  & # 0.053)

                (dataframe['cti'] < self.cti.value) & # zobrate z bbrtr -0.087

                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['rsi_84'] < 60) & #rsi check
                (dataframe['rsi_112'] < 60) &  #rsi check

                (dataframe['deadfish'] == 0 ) & 
                (dataframe['deadfish2'] == 0 ) &
                (dataframe['deadfish3'] == 0 ) &
                (dataframe['deadfish4'] == 0 ) &

                (dataframe['volume'] > 0) 









           ),
        ['enter_long', 'enter_tag']] = (1, 'zvwap')
























        dataframe.loc[
            (
                dip_condition &
                cross_above_zero_condition &
                ema_condition &
                (dataframe['close'] < zvwap)
            ),
            ['enter_long', 'enter_tag']] = (1, 'zvwap2')
        
        
        dataframe.loc[
                (        
                
                (dataframe['close'] < dataframe['zvwap']) &
                (dataframe['tcp_percent_4'] > 0.065) &

                
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &


                (dataframe['live_data_ok_1h'])&

                (dataframe['tcp_percent_72_1h'] < 0.45) &

                (dataframe['volume'] > 0)

         ),
        ['enter_long', 'enter_tag']] = (1, 'zvwap_pricedown')
        

                
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
   
            (dataframe['volume'] > 0),
            'sell'
        ] = 0

        return dataframe
    

    
   
   
   
    initial_safety_order_trigger = -0.018
    max_safety_orders = 6#8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4
    
    
    
    
    def top_percent_change_dca(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']
        
    
 
  
       


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None


        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()





        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        
        if count_of_buys > 0:
                    last_order = filled_buys[-1]
                    if last_order.order_filled_date is not None:

                        tz = last_order.order_filled_date.tzinfo
                        current_time_with_tz = current_time.replace(tzinfo=tz)
                        time_difference = current_time_with_tz - last_order.order_filled_date
                        if time_difference.total_seconds() < 300:
                            logger.info(f"Skipping SO buy for {trade.pair} because 300 sec have not passed since the last filled buy order.")
                            return None 


        if last_candle['pump_strength'] < -0.15  :
            logger.info(f"DCA for {trade.pair} waiting for pump_strength ({last_candle['pump_strength']}) to rise above -0.15")
            return None
        
        if count_of_buys > 1 and last_candle['rsi_fast_avg'] < 20 :
            logger.info(f"DCA for {trade.pair} waiting for rsi_fast ({last_candle['rsi_fast_avg']}) to rise above 20")
            return        
        
        if count_of_buys > 2 and last_candle['smi_trend_15m'] < -1   :
           logger.info(f"DCA for {trade.pair} waiting for smi_trend 1+ actual trend ({last_candle['smi_trend_15m']})")
           return None  

























        
        
        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
            if (self.safety_order_step_scale > 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
            elif (self.safety_order_step_scale < 1):
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale,(count_of_buys - 1))) / (1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:

                    stake_amount = filled_buys[0].cost

                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale,(count_of_buys - 1))
                    amount = stake_amount / current_rate

                    return stake_amount
                except Exception as exception:

                    return None

        return None

def range_filter(dataframe, length = 10, min_rate_of_change=0.01):
        df = dataframe[['date','open','high','low','close','volume']].copy()
        highest_high = df['high'].rolling(length).max()
        lowest_low = df['low'].rolling(length).min()

        df['range'] = ((highest_high - lowest_low) / lowest_low)

        return df['range']

def smi_trend(df: DataFrame, k_length=9, d_length=3, smoothing_type='EMA', smoothing=10):
    """     
    Stochastic Momentum Index (SMI) Trend Indicator 
        
    SMI > 0 and SMI > MA: (2) Bull
    SMI < 0 and SMI > MA: (1) Possible Bullish Reversal

    SMI > 0 and SMI < MA: (-1) Possible Bearish Reversal
    SMI < 0 and SMI < MA: (-2) Bear
        
    Returns:
        pandas.Series: New feature generated 
    """
    
    ll = df['low'].rolling(window=k_length).min()
    hh = df['high'].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df['close'] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
    
    if smoothing_type == 'SMA':
        smi_ma = ta.SMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'EMA':
        smi_ma = ta.EMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'WMA':
        smi_ma = ta.WMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'DEMA':
        smi_ma = ta.DEMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'TEMA':
        smi_ma = ta.TEMA(smi, timeperiod=smoothing)
    else:
        raise ValueError("Choose an MA Type: 'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA'")

    conditions = [
        (np.greater(smi, 0) & np.greater(smi, smi_ma)), # (2) Bull 
        (np.less(smi, 0) & np.greater(smi, smi_ma)),    # (1) Possible Bullish Reversal
        (np.greater(smi, 0) & np.less(smi, smi_ma)),    # (-1) Possible Bearish Reversal
        (np.less(smi, 0) & np.less(smi, smi_ma))        # (-2) Bear
    ]

    smi_trend = np.select(conditions, [2, 1, -1, -2])

    return smi, smi_ma, smi_trend