# memecoin_aggressive_strategy.py
from freqtrade.strategy import IStrategy, Decimal, TAindicators, informative
from typing import Dict, List
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class MemecoinAggressiveStrategy(IStrategy):
    # Основные параметры
    timeframe = '5m'
    minimal_roi = {
        "0": 0.15,   # 15% прибыли в любой момент
        "10": 0.10,  # 10% после 10 свечей
        "20": 0.05   # 5% после 20 свечей
    }
    
    stoploss = -0.25  # -25% стоп-лосс
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True
    
    # Параметры для агрессивной торговли
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    
    # Динамический расчет позиции
    def custom_stake_amount(self, pair: str, current_time, current_rate, proposed_stake, min_stake, max_stake, **kwargs):
        # 20% капитала на сделку + пирамидинг
        return proposed_stake * 0.20

    # Индикаторы
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Волатильность (ATR)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Импульс (RSI + Stochastic)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        stoch = ta.STOCHF(dataframe, fastk_period=14)
        dataframe['stoch_rsi'] = stoch['fastk']
        
        # Объем (сравнение с медианным)
        median_volume = dataframe['volume'].rolling(24).median()
        dataframe['volume_ratio'] = dataframe['volume'] / median_volume
        
        # Тренд (EMA 50/200)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        
        # Фильтр мемкоинов (резкие движения)
        dataframe['price_change'] = dataframe['close'].pct_change(periods=3) * 100
        
        return dataframe

    # Сигналы входа
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Условия для лонга
        dataframe.loc[
            (
                (dataframe['volume_ratio'] > 2.0) &                  # Объем выше медианного
                (dataframe['price_change'] > 15) &                    # Резкий рост цены
                (dataframe['close'] > dataframe['ema50']) &           # Цена выше EMA50
                (dataframe['stoch_rsi'] < 80) &                       # Не в зоне перекупленности
                (dataframe['rsi'] < 70) &
                (dataframe['atr'].pct_change() > 0.1)                 # Волатильность растет
            ),
            'enter_long'] = 1

        # Условия для шорта
        dataframe.loc[
            (
                (dataframe['volume_ratio'] > 2.0) &                  # Высокий объем
                (dataframe['price_change'] < -15) &                  # Резкое падение
                (dataframe['close'] < dataframe['ema50']) &          # Цена ниже EMA50
                (dataframe['stoch_rsi'] > 20) &                      # Не в зоне перепроданности
                (dataframe['rsi'] > 30) &
                (dataframe['atr'].pct_change() > 0.1)                # Волатильность растет
            ),
            'enter_short'] = 1

        return dataframe

    # Сигналы выхода
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Динамический выход по ATR
        dataframe.loc[
            (
                (dataframe['rsi'] > 85) | 
                (dataframe['stoch_rsi'] > 90) |
                (dataframe['volume_ratio'] < 0.5)
            ),
            'exit_long'] = 1
            
        dataframe.loc[
            (
                (dataframe['rsi'] < 15) | 
                (dataframe['stoch_rsi'] < 10) |
                (dataframe['volume_ratio'] < 0.5)
            ),
            'exit_short'] = 1
            
        return dataframe

    # Пивот-менеджмент
    def adjust_trade_position(self, trade, current_time, current_rate, current_profit, min_stake, max_stake, **kwargs):
        # Усреднение при просадке 10%
        if current_profit < -0.10:
            return trade.stake_amount * 0.5  # +50% к позиции

    # Защита от просадки
    def custom_exit(self, trade, current_time, current_rate, current_profit, **kwargs):
        # Аварийный выход при резком падении объема
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if last_candle['volume_ratio'] < 0.3:
            return 'panic_sell'
            
    # Оптимизация под мемкоины
    @property
    def protections(self):
        return []  # Отключаем все защиты