import logging
import math
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import (DecimalParameter, IntParameter, RealParameter,
                                informative, merge_informative_pair, stoploss_from_open)
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series

import freqtrade.vendor.qtpylib.indicators as qtpylib

logger = logging.getLogger(__name__)

# =============================================================================
#                           INDICATOR / UTILITY FUNCTIONS
# =============================================================================

def ewo(dataframe: DataFrame, sma1_length: int = 5, sma2_length: int = 35) -> Series:
    """
    Elliot Wave Oscillator (EWO) calcolato come:
    (EMA(sma1_length) - EMA(sma2_length)) / close * 100
    """
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    return (sma1 - sma2) / dataframe['close'] * 100


def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """
    Calcolo del Williams %R:
    Mostra la posizione del close rispetto al range [low, high] degli ultimi 'period' candle.
    """
    highest_high = dataframe['high'].rolling(window=period).max()
    lowest_low = dataframe['low'].rolling(window=period).min()
    wr = (highest_high - dataframe['close']) / (highest_high - lowest_low) * -100
    return wr


def chaikin_money_flow(dataframe: DataFrame, n: int = 20, fillna: bool = False) -> Series:
    """
    Chaikin Money Flow (CMF).
    Misura l'ammontare di Money Flow Volume su un periodo 'n'.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) \
          / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # evitiamo division by zero
    mfv *= dataframe['volume']
    cmf = mfv.rolling(n, min_periods=0).sum() / dataframe['volume'].rolling(n, min_periods=0).sum()
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cmf, name='cmf')


def bollinger_bands(stock_price: Series, window_size: int, num_of_std: float):
    """
    Restituisce la banda mediana (media mobile) e la banda inferiore
    di Bollinger con deviazione standard = num_of_std.
    """
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars: DataFrame) -> Series:
    """
    Calcola il prezzo tipico delle candele Heikin Ashi:
    (ha_high + ha_low + ha_close) / 3
    """
    return (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.0


def pmax(df: DataFrame, period: int, multiplier: int, length: int, MAtype: int, src: int):
    """
    Calcolo del Pmax, una variante di SuperTrend.
    - period: ATR period
    - multiplier: fattore di moltiplicazione dell'ATR (diviso per 10 nel codice, es. 27 => 2.7)
    - length, MAtype: parametri per la media mobile
    - src: definisce su quali prezzi calcolare la MA
    """
    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    # Determina la "fonte" su cui calcolare la media (close, (high+low)/2, (H+L+O+C)/4, ecc.)
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    else:
        # default a src=3 => (H+L+O+C)/4
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    # Calcolo della media in base a MAtype
    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        # talib astratto non ha T3 nativo (dipende dalle versioni)
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(masrc, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)
    else:
        mavalue = ta.EMA(masrc, timeperiod=length)  # default

    # Calcolo ATR
    df['atr'] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier / 10) * df['atr'])
    df['basic_lb'] = mavalue - ((multiplier / 10) * df['atr'])

    basic_ub = df['basic_ub'].values
    basic_lb = df['basic_lb'].values
    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))

    for i in range(period, len(df)):
        final_ub[i] = (basic_ub[i] if (basic_ub[i] < final_ub[i - 1] or mavalue[i - 1] > final_ub[i - 1])
                       else final_ub[i - 1])
        final_lb[i] = (basic_lb[i] if (basic_lb[i] > final_lb[i - 1] or mavalue[i - 1] < final_lb[i - 1])
                       else final_lb[i - 1])

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.zeros(len(df))
    for i in range(period, len(df)):
        if pm_arr[i - 1] == final_ub[i - 1] and mavalue[i] <= final_ub[i]:
            pm_arr[i] = final_ub[i]
        elif pm_arr[i - 1] == final_ub[i - 1] and mavalue[i] > final_ub[i]:
            pm_arr[i] = final_lb[i]
        elif pm_arr[i - 1] == final_lb[i - 1] and mavalue[i] >= final_lb[i]:
            pm_arr[i] = final_lb[i]
        elif pm_arr[i - 1] == final_lb[i - 1] and mavalue[i] < final_lb[i]:
            pm_arr[i] = final_ub[i]
        else:
            pm_arr[i] = 0.0

    pm_series = pd.Series(pm_arr, index=df.index)

    # Mark the trend direction up/down
    pmx = np.where(
        (pm_arr > 0.0),
        np.where(
            mavalue < pm_arr,
            'down',
            'up'
        ),
        np.NaN
    )
    return pm_series, pmx


def VWAPB(dataframe: DataFrame, window_size: int = 20, num_of_std: float = 1):
    """
    Calcolo di VWAP + bande usando la rolling_std sulla serie della vwap.
    """
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def top_percent_change_dca(dataframe: DataFrame, length: int) -> Series:
    """
    Calcola la variazione percentuale del current close dal massimo 'Open' 
    nell'ultimo 'length' periodi:
        (rollingMax(Open, length) - Close) / Close
    Se length == 0, si limita a (Open - Close) / Close sul candle corrente.
    Restituisce una Serie con tali valori.
    """
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


def is_support(row_data: pd.Series) -> bool:
    """
    Esempio di funzione usata per determinare un 'support' su 5 candele
    (la logica la gestisci come preferisci).
    """
    conditions = []
    for row in range(len(row_data) - 1):
        if row < len(row_data) / 2:
            conditions.append(row_data[row] > row_data[row + 1])
        else:
            conditions.append(row_data[row] < row_data[row + 1])
    return reduce(lambda x, y: x & y, conditions)


# =============================================================================
#                          PLACEHOLDER PER ALTRI INDICATORI
# =============================================================================

def VIDYA(df: DataFrame, length: int = 10):
    """
    Placeholder per la VIDYA (Variable Index Dynamic Average).
    Se la usi realmente, definisci l'algoritmo o importa una libreria che la calcola.
    """
    return ta.SMA(df['close'], timeperiod=length)

def vwma(df: DataFrame, length: int = 20):
    """
    Placeholder per la VWMA (Volume Weighted Moving Average).
    """
    return df['close'].rolling(length).apply(
        lambda x: np.average(x, weights=df['volume'].loc[x.index])
    )

def zema(df: DataFrame, period: int = 9):
    """
    Placeholder per la Zero Lag EMA.
    """
    return ta.EMA(df['close'], timeperiod=period)

# =============================================================================
#                               LA STRATEGIA
# =============================================================================

class GeneStrategy(IStrategy):
    """
    Strategia di esempio con molteplici segnali di acquisto e DCA.
    Include logica di trailing stop / vendite custom e parametri ottimizzabili,
    con correzione del calcolo delle percentuali per la parte di trailing.
    """

    def version(self) -> str:
        return "2025-01-15-corrected"

    # ROI table
    minimal_roi = {
        "0": 100
    }

    # Stoploss
    stoploss = -0.99  # uso custom stoploss -> vedi self.custom_stoploss

    # DCA
    position_adjustment_enable = True

    # Trailing stop
    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': "market",
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # -------------------------------------------------------------------------
    #                          PARAMETRI OTTIMIZZABILI
    # -------------------------------------------------------------------------

    fast_ewo = 50
    slow_ewo = 200

    # NFINext44
    buy_44_ma_offset = 0.982
    buy_44_ewo = -18.143
    buy_44_cti = -0.8
    buy_44_r_1h = -75.0

    # NFINext37
    buy_37_ma_offset = 0.98
    buy_37_ewo = 9.8
    buy_37_rsi = 56.0
    buy_37_cti = -0.7

    # NFINext7
    buy_ema_open_mult_7 = 0.030
    buy_cti_7 = -0.89

    # Parametri a scopo dimostrativo
    buy_rmi = IntParameter(30, 50, default=45, space='buy', optimize=True)
    buy_cci = IntParameter(-135, -90, default=-126, space='buy', optimize=True)
    buy_srsi_fk = IntParameter(30, 50, default=42, space='buy', optimize=True)
    buy_cci_length = IntParameter(25, 45, default=42, space='buy', optimize=True)
    buy_rmi_length = IntParameter(8, 20, default=11, space='buy', optimize=True)

    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.097, space='buy', optimize=True)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.028, space='buy', optimize=True)

    buy_roc_1h = IntParameter(-25, 200, default=13, space='buy', optimize=True)
    buy_bb_width_1h = DecimalParameter(0.3, 2.0, default=1.3, space='buy', optimize=True)

    # ClucHA
    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.0005, 0.02, default=0.001, space='buy', optimize=True)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.0, default=1.0, space='buy', optimize=True)
    buy_clucha_close_bblower = DecimalParameter(0.0005, 0.02, default=0.008, space='buy', optimize=True)
    buy_clucha_closedelta_close = DecimalParameter(0.0005, 0.02, default=0.014, space='buy', optimize=True)
    buy_clucha_rocr_1h = DecimalParameter(0.5, 1.0, default=0.51, space='buy', optimize=True)

    # Local Uptrend
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.026, space='buy', optimize=True)
    buy_bb_factor = DecimalParameter(0.99, 0.999, default=0.995, space='buy', optimize=True)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=13.1, space='buy', optimize=True)

    # Buy generali
    rocr_1h = DecimalParameter(0.5, 1.0, default=0.51, space='buy', optimize=True)
    rocr1_1h = DecimalParameter(0.5, 1.0, default=0.59, space='buy', optimize=True)
    bbdelta_close = DecimalParameter(0.0005, 0.02, default=0.001, space='buy', optimize=True)
    closedelta_close = DecimalParameter(0.0005, 0.02, default=0.014, space='buy', optimize=True)
    bbdelta_tail = DecimalParameter(0.7, 1.0, default=1.0, space='buy', optimize=True)
    close_bblower = DecimalParameter(0.0005, 0.02, default=0.008, space='buy', optimize=True)

    # Sell params
    sell_fisher = DecimalParameter(0.1, 0.5, default=0.5, space='sell', optimize=True)
    sell_bbmiddle_close = DecimalParameter(0.97, 1.1, default=1.067, space='sell', optimize=True)

    # Deadfish
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.06, space='sell', optimize=True)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.1, space='sell', optimize=True)
    sell_deadfish_bb_factor = DecimalParameter(0.9, 1.2, default=1.2, space='sell', optimize=True)
    sell_deadfish_volume_factor = DecimalParameter(1.0, 2.5, default=1.9, space='sell', optimize=True)

    # SMAOffset
    base_nb_candles_buy = IntParameter(8, 20, default=13, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(8, 50, default=44, space='sell', optimize=True)
    low_offset = DecimalParameter(0.985, 0.995, default=0.991, space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=1.007, space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.01, 1.02, default=1.01, space='sell', optimize=True)

    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.25, space='sell', optimize=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.5, space='sell', optimize=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.08, space='sell', optimize=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', optimize=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.08, space='sell', optimize=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.07, space='sell', optimize=True)

    # hard stoploss profit
    pHSL = DecimalParameter(-0.5, -0.04, default=-0.163, space='sell', optimize=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.02, default=0.01, space='sell', optimize=True)
    pSL_1 = DecimalParameter(0.008, 0.02, default=0.008, space='sell', optimize=True)
    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.04, 0.1, default=0.072, space='sell', optimize=True)
    pSL_2 = DecimalParameter(0.02, 0.07, default=0.054, space='sell', optimize=True)

    # -------------------------------------------------------------------------
    #                       INFORMATIVA SU ALTRI TIMEFRAME
    # -------------------------------------------------------------------------
    def informative_pairs(self):
        """
        Definisce tutte le coppie/timeframe che vogliamo usare come dati informativi.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        # Aggiungiamo anche BTC/USDT a 5m, se necessario.
        informative_pairs += [("BTC/USDT", "5m")]
        return informative_pairs

    # -------------------------------------------------------------------------
    #                          POPULATE INDICATORS
    # -------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calcolo di tutti gli indicatori necessari alla strategia.
        """
        # ===================================================
        # 1) Otteniamo i dati informativi su BTC a 5m
        info_tf = '5m'
        informative_btc = self.dp.get_pair_dataframe('BTC/USDT', timeframe=info_tf)
        # shift(1) per evitare dati "futuri"
        informative_btc = informative_btc.copy().shift(1)

        dataframe['btc_close'] = informative_btc['close']
        dataframe['btc_ema_fast'] = ta.EMA(informative_btc, timeperiod=20)
        dataframe['btc_ema_slow'] = ta.EMA(informative_btc, timeperiod=25)
        dataframe['down'] = (dataframe['btc_ema_fast'] < dataframe['btc_ema_slow']).astype('int')

        # ===================================================
        # 2) Calcolo di varie medie e volume
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        # Bollinger bands su 20 periodi, std=2
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2']

        # Bollinger 40
        bollinger2_40 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        # EMA
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # HeikinAshi
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # ClucHA
        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()

        # SRSI
        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        # Altre bande BB
        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['bb_lowerband'] = lower
        dataframe['bb_middleband'] = mid

        # is DIP
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        dataframe['bb_delta'] = (dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2']

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        # VWAP
        vwap_low, vwap_mid, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_middleband'] = vwap_mid
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_width'] = ((vwap_high - vwap_low) / vwap_mid) * 100
        dataframe['ema_vwap_diff_50'] = (dataframe['ema_50'] - dataframe['vwap_lowerband']) / dataframe['ema_50']

        # DIP protection
        dataframe['tpct_change_0'] = top_percent_change_dca(dataframe, 0)
        dataframe['tpct_change_1'] = top_percent_change_dca(dataframe, 1)
        dataframe['tcp_percent_4'] = top_percent_change_dca(dataframe, 4)

        # EWO
        dataframe['ewo'] = ewo(dataframe, 50, 200)  # usato in NFINext44
        dataframe['EWO'] = ewo(dataframe, self.fast_ewo, self.slow_ewo)

        # EMA 16
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)

        # Ema 12/26
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        # Williams R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        # Rebuy check
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)

        # Pmax
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close']) / 4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        # Fisher RSI
        rsi = ta.RSI(dataframe)
        dataframe["fisher"] = (np.exp(2 * (0.1 * (rsi - 50))) - 1) / (np.exp(2 * (0.1 * (rsi - 50))) + 1)

        # ===================================================
        # 3) Otteniamo i dati 1h per la coppia corrente
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        inf_heikinashi = qtpylib.heikinashi(informative)
        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)
        informative['rsi_14'] = ta.RSI(informative, timeperiod=14)
        informative['cmf'] = chaikin_money_flow(informative, 20)

        sup_series = informative['low'].rolling(window=5, center=True)\
                                       .apply(lambda row: is_support(row), raw=True).shift(2)
        informative['sup_level'] = np.where(
            sup_series,
            np.where(informative['close'] < informative['open'], informative['close'], informative['open']),
            float('NaN')
        )
        informative['sup_level'] = pd.Series(informative['sup_level']).ffill()

        informative['roc'] = ta.ROC(informative, timeperiod=9)
        informative['r_480'] = williams_r(informative, period=480)

        # Bollinger bands (1h)
        bb_1h = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_lowerband2'] = bb_1h['lower']
        informative['bb_middleband2'] = bb_1h['mid']
        informative['bb_upperband2'] = bb_1h['upper']
        informative['bb_width'] = (informative['bb_upperband2'] - informative['bb_lowerband2']) / informative['bb_middleband2']

        informative['r_84'] = williams_r(informative, period=84)
        informative['cti_40'] = pta.cti(informative["close"], length=40)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        # merge informativo
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        return dataframe

    # -------------------------------------------------------------------------
    #                        POPULATE ENTRY TREND (BUY)
    # -------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Logica di acquisto: molti segnali, ognuno contrassegnato da un enter_tag.
        """
        # Check se BTC ha fatto un "dump" nelle ultime 24 candele
        btc_dump = (dataframe['btc_close'].rolling(24).max() >= (dataframe['btc_close'] * 1.03))

        # RSI check
        rsi_check = (dataframe['rsi_84'] < 60) & (dataframe['rsi_112'] < 60)

        # Esempio DIP signal
        dataframe.loc[
            (
                (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value) &
                (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value) &
                (dataframe['srsi_fk'] < self.buy_srsi_fk.value) &
                (dataframe['bb_delta'] > self.buy_bb_delta.value) &
                (dataframe['bb_width'] > self.buy_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) &
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value) &
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'DIP signal')

        # Break signal
        dataframe.loc[
            (
                (dataframe['bb_delta'] > self.buy_bb_delta.value) &
                (dataframe['bb_width'] > self.buy_bb_width.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000) &
                (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value) &
                (dataframe['roc_1h'] < self.buy_roc_1h.value) &
                (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Break signal')

        # Cluc_HA
        dataframe.loc[
            (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value) &
                (dataframe['bb_lowerband2_40'].shift() > 0) &
                (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value) &
                (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value) &
                (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value) &
                (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                (dataframe['close'] > (dataframe['sup_level_1h'] * 0.88)) &
                (dataframe['ha_close'] < dataframe['ha_close'].shift())
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'cluc_HA')

        # NFIX39
        dataframe.loc[
            (
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)) &
                (dataframe['ema_200'] > (dataframe['ema_200'].shift(48) * 1.07)) &
                (dataframe['bb_lowerband2_40'].shift().gt(0)) &
                (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)) &
                (dataframe['closedelta'].gt(dataframe['close'] * 0.01)) &
                (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)) &
                (dataframe['close'].lt(dataframe['bb_lowerband2_40'].shift())) &
                (dataframe['close'].le(dataframe['close'].shift())) &
                (dataframe['close'] > dataframe['ema_50'] * 0.912)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'NFIX39')

        # NFIX29
        dataframe.loc[
            (
                (dataframe['close'] > (dataframe['sup_level_1h'] * 0.72)) &
                (dataframe['close'] < (dataframe['ema_16'] * 0.982)) &
                (dataframe['EWO'] < -10.0) &
                (dataframe['cti'] < -0.9)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'NFIX29')

        # local_uptrend
        dataframe.loc[
            (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'local_uptrend')

        # vwap
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['tcp_percent_4'] > 0.053) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'vwap')

        # insta_signal
        dataframe.loc[
            (
                (dataframe['bb_width_1h'] > 0.131) &
                (dataframe['r_14'] < -51) &
                (dataframe['r_84_1h'] < -70) &
                (dataframe['cti'] < -0.845) &
                (dataframe['cti_40_1h'] < -0.735) &
                (btc_dump == 1)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'insta_signal')

        # NFINext44
        dataframe.loc[
            (
                (dataframe['close'] < (dataframe['ema_16'] * self.buy_44_ma_offset)) &
                (dataframe['ewo'] < self.buy_44_ewo) &
                (dataframe['cti'] < self.buy_44_cti) &
                (dataframe['r_480_1h'] < self.buy_44_r_1h) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'NFINext44')

        # NFINext37
        dataframe.loc[
            (
                (dataframe['pm'] > dataframe['pmax_thresh']) &
                (dataframe['close'] < dataframe['sma_75'] * self.buy_37_ma_offset) &
                (dataframe['ewo'] > self.buy_37_ewo) &
                (dataframe['rsi'] < self.buy_37_rsi) &
                (dataframe['cti'] < self.buy_37_cti)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'NFINext37')

        # NFINext7
        dataframe.loc[
            (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_7)) &
                ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)) &
                (dataframe['cti'] < self.buy_cti_7)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'NFINext7')

        # NFINext32
        dataframe.loc[
            (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < 46) &
                (dataframe['rsi'] > 19) &
                (dataframe['close'] < dataframe['sma_15'] * 0.942) &
                (dataframe['cti'] < -0.86)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'NFINext32')

        # sma_3
        dataframe.loc[
            (
                (dataframe['bb_lowerband2_40'].shift() > 0) &
                (dataframe['bb_delta_cluc'] > dataframe['close'] * 0.059) &
                (dataframe['ha_closedelta'] > dataframe['close'] * 0.023) &
                (dataframe['tail'] < dataframe['bb_delta_cluc'] * 0.24) &
                (dataframe['close'] < dataframe['bb_lowerband2_40'].shift()) &
                (dataframe['close'] < dataframe['close'].shift()) &
                (btc_dump == 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'sma_3')

        # WVAP
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['tpct_change_1'] > 0.04) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (rsi_check) &
                (btc_dump == 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'WVAP')

        return dataframe

    # -------------------------------------------------------------------------
    #                         POPULATE EXIT TREND (SELL)
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Logica di uscita "base" (non personalizzata).
        """
        dataframe.loc[
            (
                (dataframe['fisher'] > self.sell_fisher.value) &
                (dataframe['ha_high'] <= dataframe['ha_high'].shift(1)) &
                (dataframe['ha_high'].shift(1) <= dataframe['ha_high'].shift(2)) &
                (dataframe['ha_close'] <= dataframe['ha_close'].shift(1)) &
                (dataframe['ema_fast'] > dataframe['ha_close']) &
                ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband']) &
                (dataframe['volume'] > 0)
            ),
            'exit'
        ] = 0
        return dataframe

    # -------------------------------------------------------------------------
    #                               CUSTOM SELL
    # -------------------------------------------------------------------------
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        """
        Esempio di logica di vendita personalizzata, con trailing condizionale 
        e correzione del calcolo di max_profit rispetto al prezzo di apertura.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        # Calcolo del profitto massimo (decimale)
        # Se open_rate = 10 e max_rate = 10.5 => max_profit = 0.05 (cioè +5%)
        max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate

        if last_candle is not None:
            # trail_target_1
            if (
                (current_profit > self.sell_trail_profit_min_1.value) &
                (current_profit < self.sell_trail_profit_max_1.value) &
                (max_profit > (current_profit + self.sell_trail_down_1.value))
            ):
                return 'trail_target_1'

            # trail_target_2
            elif (
                (current_profit > self.sell_trail_profit_min_2.value) &
                (current_profit < self.sell_trail_profit_max_2.value) &
                (max_profit > (current_profit + self.sell_trail_down_2.value))
            ):
                return 'trail_target_2'

            # RSI-85 target
            elif (current_profit > 3.0) and (last_candle['rsi'] > 85):
                return 'RSI-85 target'

            # Sell Signal1
            if (
                (current_profit > 0) and
                (count_of_buys < 4) and
                (last_candle['close'] > last_candle['hma_50']) and
                (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) and
                (last_candle['rsi'] > 50) and
                (last_candle['volume'] > 0) and
                (last_candle['rsi_fast'] > last_candle['rsi_slow'])
            ):
                return 'sell signal1'

            # Sell Signal1 * 1.01
            if (
                (current_profit > 0) and
                (count_of_buys >= 4) and
                (last_candle['close'] > last_candle['hma_50'] * 1.01) and
                (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) and
                (last_candle['rsi'] > 50) and
                (last_candle['volume'] > 0) and
                (last_candle['rsi_fast'] > last_candle['rsi_slow'])
            ):
                return 'sell signal1 * 1.01'

            # Sell Signal2
            if (
                (current_profit > 0) and
                (last_candle['close'] > last_candle['hma_50']) and
                (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) and
                (last_candle['volume'] > 0) and
                (last_candle['rsi_fast'] > last_candle['rsi_slow'])
            ):
                return 'sell signal2'

            # Esempio di stoploss "deadfish"
            if (
                (current_profit < self.sell_deadfish_profit.value) and
                (last_candle['close'] < last_candle['ema_200']) and
                (last_candle['bb_width'] < self.sell_deadfish_bb_width.value) and
                (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value) and
                (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value) and
                (last_candle['cmf'] < 0.0)
            ):
                return "sell_stoploss_deadfish"

        return None

    # -------------------------------------------------------------------------
    #                            CUSTOM STOPLOSS
    # -------------------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss dinamico basato su soglie di profitto (pPF_1, pPF_2) e sullo scaling lineare
        del trailing tra pSL_1 e pSL_2.
        """
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Se per qualche motivo sl_profit >= current_profit, fallback
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    # -------------------------------------------------------------------------
    #                     GESTIONE DCA (adjust_trade_position)
    # -------------------------------------------------------------------------
    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        """
        Logica DCA personalizzata. Se l'attuale profitto è sopra la soglia
        initial_safety_order_trigger, non eseguiamo DCA. Altrimenti valutiamo
        le condizioni e la scalatura di volume.
        """
        if current_profit > self.initial_safety_order_trigger:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        # Evitiamo di continuare a comprare se le candele sono brutte (condizioni "waiting")
        if count_of_buys == 1 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']):
            return None
        elif count_of_buys == 2 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
            return None
        elif count_of_buys == 3 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
            return None
        elif count_of_buys == 4 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5'] >= last_candle['ema_10']):
            return None
        elif (count_of_buys == 5) and (
                (last_candle['cmf_1h'] < 0.00) and
                (last_candle['close'] < last_candle['open']) and
                (last_candle['rsi_14_1h'] < 30) and
                (last_candle['tpct_change_0'] > 0.018) and
                (last_candle['ema_vwap_diff_50'] < 0.215) and
                (last_candle['ema_5'] >= last_candle['ema_10'])
        ):
            logger.info(
                f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) > 0, rsi_1h > 30, etc."
            )
            return None
        elif (count_of_buys == 6) and (
                (last_candle['cmf_1h'] < 0.00) and
                (last_candle['close'] < last_candle['open']) and
                (last_candle['rsi_14_1h'] < 30) and
                (last_candle['tpct_change_0'] > 0.018) and
                (last_candle['ema_vwap_diff_50'] < 0.215) and
                (last_candle['ema_5'] >= last_candle['ema_10'])
        ):
            logger.info(
                f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) > 0, rsi_1h > 30, etc."
            )
            return None
        elif (count_of_buys == 7) and (
                (last_candle['cmf_1h'] < 0.00) and
                (last_candle['close'] < last_candle['open']) and
                (last_candle['rsi_14_1h'] < 30) and
                (last_candle['tpct_change_0'] > 0.018) and
                (last_candle['ema_vwap_diff_50'] < 0.215) and
                (last_candle['ema_5'] >= last_candle['ema_10'])
        ):
            logger.info(
                f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) > 0, rsi_1h > 30, etc."
            )
            return None
        elif (count_of_buys == 8) and (
                (last_candle['cmf_1h'] < 0.00) and
                (last_candle['close'] < last_candle['open']) and
                (last_candle['rsi_14_1h'] < 30) and
                (last_candle['tpct_change_0'] > 0.018) and
                (last_candle['ema_vwap_diff_50'] < 0.215) and
                (last_candle['ema_5'] >= last_candle['ema_10'])
        ):
            logger.info(
                f"DCA for {trade.pair} waiting for cmf_1h ({last_candle['cmf_1h']}) > 0, rsi_1h > 30, etc."
            )
            return None

        # Se abbiamo ancora safety order disponibili, calcoliamo lo stake
        if 1 <= count_of_buys <= self.max_safety_orders:
            # progressione step
            safety_order_trigger = abs(self.initial_safety_order_trigger)
            if self.safety_order_step_scale > 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale *
                    (math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) /
                    (self.safety_order_step_scale - 1)
                )
            elif self.safety_order_step_scale < 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale *
                    (1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) /
                    (1 - self.safety_order_step_scale)
                )

            # Se l'attuale profitto è inferiore al trigger (es: -3.6%)
            if current_profit <= -safety_order_trigger:
                try:
                    # stake iniziale
                    stake_amount = filled_buys[0].cost
                    # scale del volume
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error obtaining stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None
