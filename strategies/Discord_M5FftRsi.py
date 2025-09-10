# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import math

class M5FftRsi(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0":  0.10
    }
    stoploss = -0.075
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    ticker_interval = '1h'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 260

    plot_config = {
        'main_plot': {

        },
        'subplots': {
            "fourier": {
                'input': {'color': 'red'},
                'fourierf': {'color': 'green'},
                'fouriers': {'color': 'blue'}
            }
        }
    }

    def informative_pairs(self):
        """
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        """
         ################################################################################## 
        ##                    --> Made in ITALY JN54oe in 2021  <--                       ##
        #                     MikeFive RSI-FFT Convolution Strategy                        #
        #                     -------------------------------------                        #
        #                  Transform the Rsi signal with the FFT algorithm                 #
        #               Using 2 different convolution for generating valid signal          #
        #       --------------------------------------------------------------------       #
        #    Change the values in the section 'Variabili Generali' for different result    #
        #       Also you can add a section valid for only one pair (example below).        #
        #     Try to change the values and SHARE the best config values or every pair!!    #
        ###  -------------------------------------------------------------------------   ###
        #   Disclaimer: This strategy is experimental. I'm not responsable for money loss  #
        #        -------------------------------------------------------------------       #
        #         TEST IT BEFORE USING IT!  ONLY YOU ARE RESPONSABLE OF YOUR MONEY!        #
        #        -------------------------------------------------------------------       #
        #                                                                                  #
        #     If you like my work, feel free to donate or use one of     █▀▄ ▄▀█   █▀▀▀▀   #
        #    my referral links, that would also greatly be appreciated   █  █  █   █▄▄▄    #
        #     https://accounts.binance.com/it/register?ref=LV0YOVHW      █     █       █   #
        #        BTC: bc1q0c0a5tez6lhjatdjkcccrg7xx0z3c2f5yk8c62         ▀     ▀   ▀▀▀▀    #
        ##                                                                                ##
         ################################################################################## 

        """
        
        #Costanti
        pi = math.pi

        #Variabili Generali
        if(True):
            Nrsi = 14          # Timeperiod for RSI (standard value is 14)
            N = 22             # Sample for FFT (experiment Prime numbers)
            QtF = 16           # N of armonics for fast curve < N (if 0 no phase)
            QtS = 1            # N of armonics for slow curve < N (if 0 no phase)
            PhF = 0.3          # Phase offset - +  fast curve (Relative to RSI)
            PhS = 0.1          # Phase offset - +  slow curve (Relative to Fast)
            AmpArm0F = 1.005   # Amp of dc value   fast curve (dc no phase)
            AmpArm0S = 1.000   # Amp of dc value   slow curve (dc no phase)
            deltaT   = 1       # Delta Timeframe for signal (1 to n+)
            deltaVb  = 0.995   # ratio buy signal
            deltaVs  = 1.002   # ratio sell signal
        
        '''
        #Variabili per singole coppie
        if(metadata['pair'] == "BTC/EUR"):
            Nrsi = 15          # Timeperiod for RSI (standard value is 14)
            N = 22             # Sample for FFT (experiment Prime numbers)
            QtF = 16           # N of armonics for fast curve
            QtS = 1            # N of armonics for slow curve
            PhF = 0.0          # Phase offset - +  fast curve (not for dc)
            PhS = 0.0          # Phase offset - +  slow curve (not for dc)
            AmpArm0F = 1.006   # Amp of dc value   fast curve (no phase)
            AmpArm0S = 1.000   # Amp of dc value   slow curve (no phase)
            deltaT   = 1       # Delta Timeframe for signal (1 to n+)
            deltaVb  = 0.998   # ratio buy signal
            deltaVs  = 1.001   # ratio sell signal
        '''
        
        #Filtro in ingresso prima di fft
        #dataframe['media'] = (dataframe['open'] + dataframe['close'] + dataframe['low'] + dataframe['high']) / 4
        dataframe['input'] = ta.RSI(dataframe['close'], timeperiod=Nrsi)
        
        #Parte Reale nel Dominio di Frequenza
        def ReX(k):
            sum = 0.0
            for i in range(0, N):
                sum = sum + dataframe['input'].shift(i)*math.cos(2*pi*k*i/N)
            return sum

        #Parte Imaginaria nel Dominio di Frequenza
        def ImX(k):
            sum = 0.0
            for i in range(0, N):
                sum = sum + dataframe['input'].shift(i)*math.sin(2*pi*k*i/N)
            return -sum

        #Conversione da Dominio di Frequeza a Dominio di Tempo (Sinusoide)  
        def ReX_(k):
            case = 0.0
            if(k!=0 and k!=N/2):
                case = 2*ReX(k)/N
            if(k==0):
                case = ReX(k)/N
            if(k==N/2):
                case = ReX(k)/N
            return case

        #Conversione da Dominio di Frequeza a Dominio di Tempo (Sinusoide)  
        def ImX_(k):
            return -2*ImX(k)/N

        '''
        #FFT NON USATA
        def x(i, N):
            sum1 = 0.0
            sum2 = 0.0
            for k in range(0, N/2): #for k=0 to N/2
                sum1 = sum1 + ReX_(k)*math.cos(2*pi*k*i/N)
            for k in range(0, N/2): #for k=0 to N/2
                sum2 = sum2 + ImX_(k)*math.sin(2*pi*k*i/N)
            return sum1+sum2
        '''
        
        #DFFT Ricostruzione in singole sinusoidi (Fase,Armonica)
        def sx(i, k):
            sum1 = ReX_(k)*math.cos(2*pi*k*i/N)
            sum2 = ImX_(k)*math.sin(2*pi*k*i/N)
            return sum1+sum2
        
        #Costruzione segnale fast
        FaseF = PhF + N
        dataframe['fourierf'] = sx(FaseF,0) * AmpArm0F
        for i in range(1, QtF+1):
            dataframe['fourierf'] = dataframe['fourierf'] + sx(FaseF,i)

        #Costruzione segnale slow
        FaseS = PhS + FaseF
        dataframe['fouriers'] = sx(FaseS,0) * AmpArm0S
        for i in range(1, QtS+1):
            dataframe['fouriers'] = dataframe['fouriers'] + sx(FaseS,i)

        #Regole buy long
        dataframe['compra'] = ((dataframe['fourierf'] / dataframe['fouriers']) > deltaVb) & ((dataframe['fourierf'].shift(deltaT) / dataframe['fouriers'].shift(deltaT)) <= deltaVb)
        
        #Regole sell long
        dataframe['vendi']  = ((dataframe['fourierf'] / dataframe['fouriers']) < deltaVs) & ((dataframe['fourierf'].shift(deltaT) / dataframe['fouriers'].shift(deltaT)) >= deltaVs)
        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['compra'] > 0) &
                (dataframe['compra'].shift(1) == 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['vendi'] > 0) &
                (dataframe['vendi'].shift(1) == 0)
            ),
            'sell'] = 1
        return dataframe
