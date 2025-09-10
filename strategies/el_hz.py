# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

class el(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 12,
        "ewo_high": 4.428,
        "ewo_low": -12.383,
        "low_offset": 0.915,
        "rsi_buy": 44,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 72,
        "high_offset": 1.008,
    }

    # ROI table (агрессивная настройка):
    minimal_roi = {
        "0": 0.15,    # 15% для быстрого выхода
        "10": 0.08,   # 8% через 10 свечей
        "30": 0.04,   # 4% через 30 свечей
        "120": 0      # Безубыток через 120 свечей
    }

    # Stoploss:
    stoploss = -0.242

    # Trailing stop (более агрессивный):
    trailing_stop = True
    trailing_stop_positive = 0.15
    trailing_stop_positive_offset = 0.20
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 12  # Увеличено для диверсификации
    
    # SMAOffset
    base_nb_candles_buy = IntParameter(5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    
    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    
    # Profit Protection
    profit_threshold_1 = DecimalParameter(0.03, 0.08, default=0.05, space='sell', optimize=True)
    profit_threshold_2 = DecimalParameter(0.08, 0.15, default=0.10, space='sell', optimize=True)
    
    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    
    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'
    process_only_new_candles = True
    startup_candle_count = 95
    plot_config = {'main_plot': {'ma_buy': {'color': 'orange'}, 'ma_sell': {'color': 'orange'}}}
    
    # Активируем кастомные функции защиты
    use_custom_stoploss = True
    use_custom_stake_amount = True

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def get_informative_indicators(self, metadata: dict):
        dataframe = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Рассчитываем все индикаторы
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        aroon = ta.AROON(dataframe)
        dataframe['aroonup'] = aroon['aroonup']
        dataframe['aroondown'] = aroon['aroondown']
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        keltner = qtpylib.keltner_channel(dataframe)
        dataframe['kc_upperband'] = keltner['upper']
        dataframe['kc_lowerband'] = keltner['lower']
        dataframe['kc_middleband'] = keltner['mid']
        dataframe['kc_percent'] = (dataframe['close'] - dataframe['kc_lowerband']) / (dataframe['kc_upperband'] - dataframe['kc_lowerband'])
        dataframe['kc_width'] = (dataframe['kc_upperband'] - dataframe['kc_lowerband']) / dataframe['kc_middleband']
        dataframe['uo'] = ta.ULTOSC(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['roc'] = ta.ROC(dataframe)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']

        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        hilbert = ta.HT_SINE(dataframe)
        dataframe['htsine'] = hilbert['sine']
        dataframe['htleadsine'] = hilbert['leadsine']

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        
        # Добавляем ATR для волатильности
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']
        
        # Средний объем
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        # EMA для стратегии
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Фильтры для повышения качества входов
        volume_filter = dataframe['volume'] > dataframe['volume_ma'] * 1.2
        volatility_filter = dataframe['atr_pct'] < 0.08
        trend_filter = dataframe['adx'] > 25
        
        # Условия для лонгов
        buy_conditions = []
        buy_conditions.append(
            (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) & 
            (dataframe['EWO'] > self.ewo_high.value) & 
            (dataframe['rsi'] < self.rsi_buy.value) & 
            volume_filter &
            trend_filter
        )
        buy_conditions.append(
            (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) & 
            (dataframe['EWO'] < self.ewo_low.value) & 
            volume_filter &
            volatility_filter
        )
        
        if buy_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, buy_conditions), 'enter_long'] = 1

        # Условия для шортов
        sell_conditions = []
        sell_conditions.append(
            (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value) & 
            volume_filter &
            trend_filter
        )
        
        if sell_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, sell_conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Частичный выход для фиксации прибыли
        dataframe.loc[
            (dataframe['close'] > dataframe['open'] * 1.05) & 
            (qtpylib.crossed_above(dataframe['rsi'], 70)),
            'exit_long'
        ] = 0.5  # Закрываем 50% позиции

        # Основные условия выхода
        exit_long_conditions = []
        exit_long_conditions.append(
            (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)
        )
        if exit_long_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_long_conditions), 'exit_long'] = 1

        exit_short_conditions = []
        exit_short_conditions.append(
            (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) & 
            (dataframe['EWO'] > self.ewo_high.value) & 
            (dataframe['rsi'] < self.rsi_buy.value)
        )
        exit_short_conditions.append(
            (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) & 
            (dataframe['EWO'] < self.ewo_low.value)
        )
        if exit_short_conditions:
            dataframe.loc[reduce(lambda x, y: x | y, exit_short_conditions), 'exit_short'] = 1

        return dataframe

    # Адаптивный стоп-лосс с защитой прибыли
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, 
                       current_rate: float, current_profit: float, **kwargs) -> float:
        # Жесткий стоп при убытке >12%
        if current_profit < -0.12:
            return -1
            
        # Защита прибыли
        if current_profit > self.profit_threshold_2.value:
            return -0.02  # Очень тугой стоп при +10%
        elif current_profit > self.profit_threshold_1.value:
            return -0.05  # Защищаем 5% прибыли
            
        # Базовый стоп
        return -0.242

    # Управление размером позиции (риск 3% на сделку)
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, 
                           proposed_stake: float, min_stake: float, max_stake: float, 
                           leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        total_balance = self.wallets.get_total('USDT')
        return min(proposed_stake, total_balance * 0.50)

    def leverage(self, pair: str, current_time: datetime, current_rate: float, 
                proposed_leverage: float, max_leverage: float, entry_tag: str, 
                side: str, **kwargs) -> float:
        return 10.0