from pandas import DataFrame,Series
from functools import reduce

import talib.abstract as ta
import pandas_ta
import pandas as pd
from freqtrade.strategy import IStrategy
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import technical.indicators as ftt
import numpy as np
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import zema, VIDYA
import time


class ichitest(IStrategy):
    timeframe = '5m'
        

    # Custom stoploss
    use_custom_stoploss = True
    use_sell_signal = True
    
    process_only_new_candles = True



    startup_candle_count = 200

    INTERFACE_VERSION = 2



    # Buy hyperspace params:
    buy_params = {
        "buy_T3": 11,
        "buy_T3_offset": 0.132,
        "buy_con10_enabled": True,
        "buy_con11_enabled": False,
        "buy_con12_enabled": False,
        "buy_con13_enabled": False,
        "buy_con14_enabled": False,
        "buy_con1_enabled": True,
        "buy_con2_enabled": True,
        "buy_con3_enabled": False,
        "buy_con4_enabled": False,
        "buy_con6_enabled": False,
        "buy_con7_enabled": False,
        "buy_con9_enabled": False,
        "buy_ichi_bas": 139,
        "buy_ichi_con": 26,
        "buy_ichi_dep": 72,
        "buy_ichi_lagg": 43,
        "buy_inf15m_ema": 78,
        "buy_inf1h_ema": 96,
        "buy_pmax_offset": 0.728,
        "buy_rsid": 83,
        "buy_rsik": 33,
        "buy_rsil": 44,
        "buy_stl": 85,
        "buy_stm": 1.068,
        "buy_stochd": 8,
        "buy_stochk": 3,
        "buy_stochl": 38,
        "buy_stp": 18,
    }

    # Sell hyperspace params:
    sell_params = {
        "pHSL": -0.079,
        "pPF_1": 0.008,
        "pPF_2": 0.068,
        "pSL_1": 0.019,
        "pSL_2": 0.038,
        "sell_T3": 79,
        "sell_ichi_bas": 56,
        "sell_ichi_con": 26,
        "sell_ichi_dep": 49,
        "sell_ichi_lagg": 113,
        "sell_rsid": 24,
        "sell_rsik": 73,
        "sell_rsil": 94,
        "sell_stl": 37,
        "sell_stm": 1.053,
        "sell_stochd": 1,
        "sell_stochk": 1,
        "sell_stochl": 67,
        "sell_stp": 70,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.068,
        "34": 0.043,
        "80": 0.013,
        "107": 0
    }

    # Stoploss:
    stoploss = -0.99  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy
    # Buy hyperspace params:
    buy_params = {
        "buy_T3": 11,
        "buy_T3_offset": 0.132,
        "buy_con10_enabled": True,
        "buy_con11_enabled": False,
        "buy_con12_enabled": False,
        "buy_con13_enabled": False,
        "buy_con14_enabled": False,
        "buy_con1_enabled": True,
        "buy_con2_enabled": True,
        "buy_con3_enabled": False,
        "buy_con4_enabled": False,
        "buy_con6_enabled": False,
        "buy_con7_enabled": False,
        "buy_con9_enabled": False,
        "buy_ichi_bas": 139,
        "buy_ichi_con": 26,
        "buy_ichi_dep": 72,
        "buy_ichi_lagg": 43,
        "buy_inf15m_ema": 78,
        "buy_inf1h_ema": 96,
        "buy_pmax_offset": 0.728,
        "buy_rsid": 83,
        "buy_rsik": 33,
        "buy_rsil": 44,
        "buy_stl": 85,
        "buy_stm": 1.068,
        "buy_stochd": 8,
        "buy_stochk": 3,
        "buy_stochl": 38,
        "buy_stp": 18,
    }

    # Sell hyperspace params:
    sell_params = {
        "pHSL": -0.079,
        "pPF_1": 0.008,
        "pPF_2": 0.068,
        "pSL_1": 0.019,
        "pSL_2": 0.038,
        "sell_T3": 79,
        "sell_ichi_bas": 56,
        "sell_ichi_con": 26,
        "sell_ichi_dep": 49,
        "sell_ichi_lagg": 113,
        "sell_rsid": 24,
        "sell_rsik": 73,
        "sell_rsil": 94,
        "sell_stl": 37,
        "sell_stm": 1.053,
        "sell_stochd": 1,
        "sell_stochk": 1,
        "sell_stochl": 67,
        "sell_stp": 70,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.068,
        "34": 0.043,
        "80": 0.013,
        "107": 0
    }

    # Stoploss:
    stoploss = -0.99  # value loaded from strategy

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy







    ichi_opt=False
    buy_ichi_con=IntParameter(20,150,default=14,load=True,optimize=ichi_opt)
    buy_ichi_bas=IntParameter(20,150,default=14,load=True,optimize=ichi_opt)
    buy_ichi_lagg=IntParameter(20,150,default=14,load=True,optimize=ichi_opt)
    buy_ichi_dep=IntParameter(20,150,default=14,load=True,optimize=ichi_opt)
    # buy_ichi_con_offset=DecimalParameter(0, 5, default=0.080, decimals=2,load=True,optimize=ichi_opt)

    timeframe_opt=False
    buy_inf15m_ema=IntParameter(2,100,default=14,load=True,optimize=timeframe_opt)
    buy_inf1h_ema=IntParameter(2,100,default=14,load=True,optimize=timeframe_opt)

    pmax_opt=False
    buy_stl=IntParameter(2,100,default=18,load=True,optimize=pmax_opt)
    buy_stp=IntParameter(2,100,default=44,load=True,optimize=pmax_opt)
    buy_stm=DecimalParameter(.5,4,default=3.63,decimals=3,load=True,optimize=pmax_opt)
    buy_pmax_offset=DecimalParameter(0,1.6,default=1,decimals=3,load=True,optimize=pmax_opt)
    
    t3_opt=False
    buy_T3=IntParameter(2,100,default=18,load=True,optimize=t3_opt)
    buy_T3_offset =DecimalParameter(0, 1.6, default=0.080, decimals=3, load=True,optimize=t3_opt)

    rsi_opt=False
    buy_rsil=IntParameter(2,100,default=18,load=True,optimize=rsi_opt)
    buy_stochl=IntParameter(2,100,default=18,load=True,optimize=rsi_opt)
    buy_stochk=IntParameter(1,10,default=3,load=True,optimize=rsi_opt)
    buy_stochd=IntParameter(1,10,default=3,load=True,optimize=rsi_opt)
    buy_rsik=IntParameter(2,100,default=18,load=True,optimize=rsi_opt)
    buy_rsid=IntParameter(2,100,default=18,load=True,optimize=rsi_opt)

    pmax_opt_sell=True
    sell_stl=IntParameter(2,100,default=18,load=True,optimize=pmax_opt_sell)
    sell_stp=IntParameter(2,100,default=44,load=True,optimize=pmax_opt_sell)
    sell_stm=DecimalParameter(.5,4,default=3.63,decimals=3,load=True,optimize=pmax_opt_sell)
    sell_pmax_offset=DecimalParameter(0,1.6,default=1,decimals=3,load=True,optimize=pmax_opt_sell)

    sell_T3=IntParameter(2,100,default=18,load=True,optimize=True)
    sell_T3_offset =DecimalParameter(0, 1.6, default=0.080, decimals=3, load=True,optimize=True)

    ichi_opt_sell=False
    sell_ichi_con=IntParameter(20,150,default=14,load=True,optimize=ichi_opt_sell)
    sell_ichi_bas=IntParameter(20,150,default=14,load=True,optimize=ichi_opt_sell)
    sell_ichi_lagg=IntParameter(20,150,default=14,load=True,optimize=ichi_opt_sell)
    sell_ichi_dep=IntParameter(20,150,default=14,load=True,optimize=ichi_opt_sell)

    buy_con_test=False
    buy_con1_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con2_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con3_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con4_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con6_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con7_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con9_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con10_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con11_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con12_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con13_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)
    buy_con14_enabled=CategoricalParameter([True,False],default=True,optimize=buy_con_test,load=True)



     # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True,optimize=False)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True,optimize=False)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True,optimize=False)

    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True,optimize=False)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True,optimize=False)


    sell_rsil=IntParameter(2,100,default=18,load=True,optimize=False)
    sell_stochl=IntParameter(2,100,default=18,load=True,optimize=False)
    sell_stochk=IntParameter(1,10,default=3,load=True,optimize=False)
    sell_stochd=IntParameter(1,10,default=3,load=True,optimize=False)
    sell_rsik=IntParameter(2,100,default=18,load=True,optimize=False)
    sell_rsid=IntParameter(2,100,default=18,load=True,optimize=False)


    



    inf_15m = '15m'
    info_timeframe_1h= '1h'
    



    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.info_timeframe_1h)

        return informative_1h

    # Informative indicator for pump detection 

    def informative_15m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_15m)
        

        
        return informative_15m

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '15m') for pair in pairs]
        informative_pairs.extend([(pair, self.info_timeframe_1h) for pair in pairs])

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.info_timeframe_1h)

        return dataframe
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        #Import 1h indicators
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_1h, self.timeframe, self.info_timeframe_1h, ffill=True)
        drop_columns = [(s + "_" + self.info_timeframe_1h) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        #Import 15m indicators    
        informative_15m = self.informative_15m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(
            dataframe, informative_15m, self.timeframe, self.inf_15m, ffill=True)
        drop_columns = [(s + "_" + self.inf_15m) for s in ['date']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)
        
      
        return dataframe

    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['ohcl15m']=(dataframe['open_15m']+dataframe['high_15m']+dataframe['close_15m']+dataframe['low_15m'])/4
        dataframe['ohcl1h']=(dataframe['open_1h']+dataframe['high_1h']+dataframe['close_1h']+dataframe['low_1h'])/4
        dataframe['ema15m']=pandas_ta.ema(dataframe['ohcl15m'],length=int(self.buy_inf15m_ema.value))
        dataframe['ema1h']=pandas_ta.ema(dataframe['ohcl1h'],length=int(self.buy_inf1h_ema.value))
        
        dataframe['T3']=T3(dataframe,length=int(self.buy_T3.value))
        dataframe['pm'],dataframe['ma']= pmax(dataframe,period=int(self.buy_stp.value),
        multiplier=self.buy_stm.value,length=int(self.buy_stl.value),MAtype=8,src=3)
        
        dataframe['T3sell']=T3(dataframe,length=int(self.sell_T3.value))
        dataframe['pmsell'],dataframe['masell']= pmax(dataframe,period=int(self.sell_stp.value),
        multiplier=self.sell_stm.value,length=int(self.sell_stl.value),MAtype=8,src=3)

        
        
        
        displacement = int(self.buy_ichi_dep.value)
        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=int(self.buy_ichi_con.value), 
            base_line_periods=int(self.buy_ichi_bas.value),
            laggin_span=int(self.buy_ichi_lagg.value), 
            displacement=displacement
            )

        displacement = int(self.sell_ichi_dep.value)
        sellicho = ftt.ichimoku(dataframe, 
            conversion_line_period=int(self.sell_ichi_con.value), 
            base_line_periods=int(self.sell_ichi_bas.value),
            laggin_span=int(self.sell_ichi_lagg.value), 
            displacement=displacement
        )

        # cross indicators
        dataframe['Conversion_Linesell'] = sellicho['tenkan_sen']
        dataframe['Base_Linesell'] = sellicho['kijun_sen']

        # cloud, green a > b, red a < b
        dataframe['senkou_asell'] = sellicho['senkou_span_a']
        dataframe['senkou_bsell'] = sellicho['senkou_span_b']
        dataframe['leading_senkou_span_asell'] = sellicho['leading_senkou_span_a']
        dataframe['leading_senkou_span_bsell'] = sellicho['leading_senkou_span_b']
            
        
        dataframe['chikou_span'] = ichimoku['chikou_span']

        # cross indicators
        dataframe['Conversion_Line'] = ichimoku['tenkan_sen']
        dataframe['Base_Line'] = ichimoku['kijun_sen']

        # cloud, green a > b, red a < b
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        dataframe['future_green'] = (dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']).astype('int') * 2

        

        # The chikou_span is shifted into the past, so we need to be careful not to read the
        # current value.  But if we shift it forward again by displacement it should be safe to use.
        # We're effectively "looking back" at where it normally appears on the chart.
        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['senkou_a']) &
                (dataframe['chikou_span'] > dataframe['senkou_b'])&
                (dataframe['chikou_span'] > dataframe['leading_senkou_span_a'])&
                (dataframe['chikou_span'] > dataframe['leading_senkou_span_b'])
            ).shift(displacement).fillna(0).astype('int')
        ichimoku['chikou_span']=dataframe['chikou_span']
        dataframe['curnt_srpead']=dataframe['Conversion_Line']-dataframe['Base_Line']
        dataframe['chikou_span']=ichimoku['chikou_span'].shift(26).fillna(0).astype('int')
        # dataframe['prv1_srpead']=dataframe['leading_senkou_span_a'].shift(1)-dataframe['leading_senkou_span_b'].shift(1)
        # dataframe['prv2_srpead']=dataframe['leading_senkou_span_a'].shift(2)-dataframe['leading_senkou_span_b'].shift(2)

        dataframe['prv1_srpead']=dataframe['Conversion_Line'].shift(1)-dataframe['Base_Line'].shift(1)
        dataframe['prv2_srpead']=dataframe['Conversion_Line'].shift(2)-dataframe['Base_Line'].shift(2)

        dataframe['stochrsiK'],dataframe['stochrsid']=ta.STOCHRSI(dataframe['close'],length=int(self.buy_stochl.value),
        rsi_length=int(self.buy_rsil.value),k=int(self.buy_stochk.value),d=int(self.buy_stochd.value))

        dataframe['stochrsiKsell'],dataframe['stochrsidsell']=ta.STOCHRSI(dataframe['close'],length=int(self.sell_stochl.value),
        rsi_length=int(self.sell_rsil.value),k=int(self.sell_stochk.value),d=int(self.sell_stochd.value))



        conditions = []
        if self.buy_con1_enabled:
            conditions.append((dataframe['Conversion_Line'] > dataframe['Base_Line'])) 

        if self.buy_con1_enabled.value:
            conditions.append(dataframe['close'] > dataframe['senkou_a'])
        if self.buy_con2_enabled.value:
            conditions.append(dataframe['close'] > dataframe['senkou_b']) 
        if self.buy_con3_enabled.value:
            conditions.append(dataframe['future_green'] > 0) 
        if self.buy_con4_enabled.value:
            conditions.append(dataframe['chikou_high'] > 0)    
        if self.buy_con6_enabled.value:
            conditions.append(dataframe['curnt_srpead']>dataframe['prv1_srpead'])
        if self.buy_con7_enabled.value:
            conditions.append(dataframe['prv1_srpead']>dataframe['prv2_srpead'])    
        if self.buy_con9_enabled.value:
            conditions.append(dataframe['ma']> dataframe['Conversion_Line'])
        if self.buy_con10_enabled.value:
            conditions.append(dataframe['chikou_span']> dataframe['leading_senkou_span_a'])
        if self.buy_con11_enabled.value:
            conditions.append(dataframe['chikou_span']> dataframe['leading_senkou_span_b'])
        if self.buy_con12_enabled.value:
            conditions.append(dataframe['stochrsiK']>dataframe['stochrsid'])
        if self.buy_con13_enabled.value:
            conditions.append(dataframe['stochrsiK']>self.buy_rsik.value)
        if self.buy_con14_enabled.value:
            conditions.append(dataframe['stochrsid']>self.buy_rsid.value)
        (  
           conditions.append((dataframe['close']>dataframe['ema15m'])&
           (dataframe['close']>dataframe['ema1h'])&
           (dataframe['ema15m']>dataframe['ema1h'])&
           (dataframe['ma']>dataframe['pm']*self.buy_pmax_offset.value)&
           (dataframe['close']> dataframe['T3']*self.buy_T3_offset.value)&
            dataframe['volume']>0))
        
        # if self.buy_inf1d_ema_enabled.value:
        #  conditions.append((dataframe['close']>dataframe['ema15m'])&
        #  (dataframe['close']>dataframe['ema1d'])&
        #  (dataframe['ema15m']>dataframe['ema1d']))
        
        
           
        if conditions:
             dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
        return dataframe

    

        
   
   
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(
        (dataframe['Conversion_Linesell'] < dataframe['Base_Linesell'])
        |
        (dataframe['leading_senkou_span_asell'] < dataframe['leading_senkou_span_bsell'])
        |
        (dataframe['masell']<dataframe['pmsell']*self.sell_pmax_offset.value)
        |
        # (dataframe['masell']< dataframe['Conversion_Linesell'])
        # |
        (dataframe['stochrsiKsell']<dataframe['stochrsidsell'])
        |
        (dataframe['close']< dataframe['T3']*self.sell_T3_offset.value)
        |
        (dataframe['stochrsiK']< self.sell_rsik.value)
        |
        (dataframe['stochrsid']< self.sell_rsid.value)
         )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        return dataframe
    
def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = ftt.VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, mavalue


def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']

        