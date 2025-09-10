
from freqtrade.strategy import IStrategy
import pandas as pd
import ta  # Import the 'ta' module for technical indicators
import numpy as np

class candlestickSD(IStrategy):
     timeframe = '1m'
     tol = 0.003 
     A=0.3
     B=2
     C=2
     cross_candle_length = 5  # Minimum candles for a cross
     cross_tolerance = 0.003  # Cross tolerance
     take_profit = 0.0075  # Take profit percentage
     stop_loss = -0.005  # Stop loss percentage
     stoploss = stop_loss
     def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        Tol=dataframe['close'][1]*0.003
        dataframe['Candle'] = 0
        dataframe['Type'] = 0
        dataframe['SD']=0
        SD=[]
        for i in range(len(dataframe)):
            if abs(dataframe['close'][i]-dataframe['high'][i])<(self.A):
                if abs(dataframe['open'][i]-dataframe['low'][i])<self.A*Tol:
                    dataframe['Candle'][i] = 'Spike'
                    dataframe['Type'][i] = 'G'
            if abs(dataframe['close'][i]-dataframe['low'][i])<self.A*Tol:
                if abs(dataframe['open'][i]-dataframe['high'][i])<self.A*Tol:
                    dataframe['Candle'][i] = 'Spike'
                    dataframe['Type'][i] = 'R'
            if abs(dataframe['close'][i]-dataframe['open'][i])<self.A*Tol:
                if (dataframe['high'][i]-dataframe['low'][i])>Tol:
                    dataframe['Candle'][i] = 'Doji'
                    if dataframe['close'][i]>dataframe['open'][i]:
                        dataframe['Type'][i] = 'G'
                    else:
                        dataframe['Type'][i] = 'R'
            if abs(dataframe['close'][i]-dataframe['high'][i])<self.B*self.A*Tol:
                if (dataframe['high'][i]-dataframe['low'][i])>Tol:
                    
                    dataframe['Candle'][i]='Hammer'
                    if dataframe['close'][i]>dataframe['open'][i]:
                        dataframe['Type'][i] = 'G'
                    else:
                        dataframe['Type'][i] = 'R'
            if abs(dataframe['open'][i]-dataframe['high'][i])<self.B*self.A*Tol:
                if (dataframe['high'][i]-dataframe['low'][i])>Tol:
                    dataframe['Candle'][i]='Hammer'
                    if dataframe['close'][i]>dataframe['open'][i]:
                        dataframe['Type'][i] = 'G'
                    else:
                        dataframe['Type'][i] = 'R'
            if abs(dataframe['close'][i]-dataframe['low'][i])<self.B*self.A*Tol:
                if (dataframe['high'][i]-dataframe['low'][i])>Tol:
                    dataframe['Candle'][i]='InHammer'
                    if dataframe['close'][i]>dataframe['open'][i]:
                        dataframe['Type'][i] = 'G'
                    else:
                        dataframe['Type'][i] = 'R'
            if abs(dataframe['open'][i]-dataframe['low'][i])<self.B*self.A*Tol:
                if (dataframe['high'][i]-dataframe['low'][i])>Tol:
                    dataframe['Candle'][i]='InHammer'
                    if dataframe['close'][i]>dataframe['open'][i]:
                        dataframe['Type'][i] = 'G'
                    else:
                        dataframe['Type'][i] = 'R'
        for i in range(24*60,len(dataframe)):
            SD=[]
            A1=np.max(dataframe['close'][i-15:i])
            A2=np.min(dataframe['close'][i-15:i])
            A3=np.max(dataframe['close'][i-60:i])
            A4=np.min(dataframe['close'][i-60:i])
            A5=np.max(dataframe['close'][i-240:i])
            A6=np.min(dataframe['close'][i-240:i])
            A7=np.max(dataframe['close'][i-24*60:i])
            A8=np.min(dataframe['close'][i-24*60:i])
            B1=np.argmax(dataframe['close'][i-15:i])
            B2=np.argmin(dataframe['close'][i-15:i])
            B3=np.argmax(dataframe['close'][i-60:i])
            B4=np.argmin(dataframe['close'][i-60:i])
            B5=np.argmax(dataframe['close'][i-240:i])
            B6=np.argmin(dataframe['close'][i-240:i])
            B7=np.argmax(dataframe['close'][i-24*60:i])
            B8=np.argmin(dataframe['close'][i-24*60:i])
         
            if 24*60-B7>64*self.C and B7>64*self.C:
                if i==1:
                    SDt.append(A7)
                    dataframe['SD'][i-24*60+B7]=1
                else:
                    SD.append(A7)
                    dataframe['SD'][i-24*60+B7]=1
            if 24*60-B8>64*self.C and B8>64*self.C:
                if i==1:
                    SDt.append(A8)
                    dataframe['SD'][i-24*60+B8]=1
                else:
                    SD.append(A8)   
                    dataframe['SD'][i-24*60+B8]=1        
            if 240-B5>16*self.C and B5>16*self.C:
                SD.append(A5)
                dataframe['SD'][i-240+B5]=2
            if 240-B6>16*self.C and B6>16*self.C:
                SD.append(A6)
                dataframe['SD'][i-240+B6]=2
            if 60-B3>4*self.C and B3>4*self.C:
                SD.append(A3)
                dataframe['SD'][i-60+B3]=3
            if 60-B4>4*self.C and B4>4*self.C:
                SD.append(A4)
                dataframe['SD'][i-60+B4]=3
            if 15-B1>self.C and B1>self.C:
                SD.append(A1)
                dataframe['SD'][i-15+B1]=4
            if 15-B2>self.C and B2>self.C:
                SD.append(A2)
                dataframe['SD'][i-15+B2]=4
            
        return dataframe

     def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        SD = [index for index, arr in enumerate(dataframe['SD']) if np.any(arr != 0)]
        SDt=[]
        for j in range(len(SD)):
                flag=0
                for k in range(len(SDt)):
                    if abs(SD[j]-SDt[k])<self.tol:
                        flag=1
                if flag==0:
                    SDt.append(SD[j])
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0


        #crossUp1 = True if (dataframe['Median1'].shift(1) < dataframe['Median2'].shift(1)) & (dataframe['Median1'] >= dataframe['Median2']) else False
    #        crossDown1 = True if (dataframe['Median1'].shift(1) > dataframe['Median2'].shift(1)) & (dataframe['Median1'] <= dataframe['Median2']) else False
    #        crossUp2 = True if (dataframe['Median1'].shift(1) < dataframe['Median3'].shift(1)) & (dataframe['Median1'] >= dataframe['Median3']) else False
    #        crossDown2 = True if (dataframe['Median1'].shift(1) > dataframe['Median3'].shift(1)) & (dataframe['Median1'] <= dataframe['Median2']) else False
    #        
    #        cross_tolerance = abs(dataframe['Median1'][i] - dataframe['Median2'][i]) / dataframe['Median1'][i]
    #        buySignal = dataframe['close'].shift(1) < dataframe['Median1'].shift(1) & (dataframe['close'] >= dataframe['Median1'].shift(1)) & rsi_value > (rsi_value.shift(self.cross_candle_length).where(crossUp1).max()) &  dataframe['close']< (rsi_value.shift(self.cross_candle_length).where(crossUp1).max())
                                                                                                  
   




        for i in range(1,len(dataframe)):
            
            if dataframe['Candle'][i]=='Doji':
                print ("Long Signal Detected")
                dataframe.loc[i + self.cross_candle_length - 1, 'enter_long'] = 1
            elif dataframe['Candle'][i]=='Hammer':
                print ("Short Signal Detected")
                dataframe.loc[i + self.cross_candle_length - 1, 'enter_short'] = 1
            
        return dataframe

     def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        tolerance=0.003
        for idx, row in dataframe.iterrows():
            if row['enter_long'] == 1:
             
                entry_price = row['close']
             
                profit_target_price = entry_price * (1 + self.take_profit)
                stop_loss_price = entry_price * (1 + self.stop_loss)
            
                # Find the index of prices that are higher than entry price
                higher_prices_idx = dataframe[dataframe['close'] > profit_target_price].index
                higher_prices_idx1= dataframe[dataframe['close'] < stop_loss_price].index
                
                if higher_prices_idx.empty and higher_prices_idx1.empty:
                    continue
                
                F=higher_prices_idx-idx
                F1=higher_prices_idx1-idx
                
                # Find the index of the price closest to the profit target price among higher prices
                T1=F[F > 0].min()+idx
                T2=F1[F1 > 0].min()+idx
                m=1
                n=1
                flag=1
                if np.isnan(T1):
                    flag=0
                    m=0
                if np.isnan(T2):
                    flag=0
                    n=0
                if flag==0:
                    if m==0 and n!=0:
                        exit_idx=T2
                    elif n==0 and m!=0:
                        exit_idx=T1
                    elif m==0 and n==0:
                        exit_idx=idx+self.cross_candle_length
                else:
                    exit_idx=min(T1,T2)
                if exit_idx>len(dataframe):
                    exit_idx=len(dataframe)
                    
                if exit_idx<len(dataframe):
                    if dataframe['close'][exit_idx] >= profit_target_price :   
                        dataframe.loc[exit_idx, 'exit_long'] = 1
                    
                        print(f"Long Exit signal Detected with Profit at index {idx}")
                    elif dataframe['close'][exit_idx] <= stop_loss_price :
                        dataframe.loc[idx, 'exit_long'] = 1
                        print(f"Long Exit signal Detected with StopLoss at index {idx}")

            elif row['enter_short'] == 1:
                entry_price = row['close']
                profit_target_price = entry_price * (1 - self.take_profit)
                stop_loss_price = entry_price * (1 - self.stop_loss)
                # Find the index of prices that are lower than entry price
                lower_prices_idx = dataframe[dataframe['close'] < profit_target_price].index
                lower_prices_idx1= dataframe[dataframe['close'] > stop_loss_price].index
                
                if lower_prices_idx.empty and lower_prices_idx1.empty:
                    continue
                
                F=lower_prices_idx-idx
                F1=lower_prices_idx1-idx
                
                # Find the index of the price closest to the profit target price among higher prices
                T1=F[F > 0].min()+idx
                T2=F1[F1 > 0].min()+idx
                
                m=1
                n=1
                flag=1
                if np.isnan(T1):
                    flag=0
                    m=0
                if np.isnan(T2):
                    flag=0
                    n=0
                if flag==0:
                    if m==0 and n!=0:
                        exit_idx=T2
                    elif n==0 and m!=0:
                        exit_idx=T1
                    elif m==0 and n==0:
                        exit_idx=idx+self.cross_candle_length
                else:
                    exit_idx=min(T1,T2)
                
                if exit_idx<len(dataframe):
                    if dataframe['close'][exit_idx] <= profit_target_price:
                        dataframe.loc[exit_idx, 'exit_short'] = 1
                        print(f"Short Exit signal Detected with Profit at index {exit_idx}")
                    elif dataframe['close'][exit_idx] >= stop_loss_price :
                        dataframe.loc[idx, 'exit_long'] = 1
                        print(f"Short Exit signal Detected with StopLoss at index {idx}")

        return dataframe