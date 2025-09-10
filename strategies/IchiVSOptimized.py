# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
import numpy as np
from freqtrade.strategy import merge_informative_pair, stoploss_from_open
from typing import Optional

class IchiVSOptimized(IStrategy):
    """
    IchiVS_Optimized - Chiến lược cải tiến cho giao dịch cả long và short trên KuCoin Spot.
    
    Đặc điểm chính:
      - Sử dụng Heikin Ashi để làm mịn dữ liệu giá mà không ghi đè lên OHLC gốc (tạo thêm cột riêng).
      - Tính toán EMA đa khung thời gian (5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h) để xác định xu hướng.
      - Dùng Ichimoku Cloud để lọc xu hướng (long: giá trên mây, short: giá dưới mây).
      - Sử dụng Fan Magnitude (tỉ lệ giữa EMA 1h và EMA 8h) cùng với xu hướng tăng/giảm để xác nhận sức mạnh xu hướng.
      - Tín hiệu thoát dựa trên giao cắt giữa giá 5m và EMA 2h, có thể bổ sung trailing stop dựa trên ATR.
    """
    timeframe = '5m'
    startup_candle_count = 96
    process_only_new_candles = True

    can_short = True
    leverage_value = 5

    # ROI & Stoploss
    minimal_roi = {
        "0": 0.05,
        "10": 0.03,
        "30": 0.015,
        "60": 0
    }
    stoploss = -0.275

    trailing_stop = False  # Có thể bật trailing stop dựa trên ATR nếu cần
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # --- Buy (Long) Parameters ---
    buy_params = {
        "buy_trend_above_senkou_level": 2,    # Yêu cầu tín hiệu Ichimoku từ khung 5m và 15m
        "buy_trend_bullish_level": 3,           # Yêu cầu bullish từ khung 5m, 15m, 30m
        "buy_fan_magnitude_shift_value": 2,
        "buy_min_fan_magnitude_gain": 1.002,    # Yêu cầu Fan Magnitude tăng và > 1
        # Có thể thêm tham số cho bộ lọc ATR:
        "atr_threshold": 0.0015,
    }

    # --- Short Parameters ---
    short_params = {
        "short_trend_below_senkou_level": 2,    # Yêu cầu tín hiệu Ichimoku từ khung 5m và 15m (giá dưới mây)
        "short_trend_bearish_level": 3,           # Yêu cầu bearish từ khung 5m, 15m, 30m
        "short_fan_magnitude_shift_value": 2,
        "short_max_fan_magnitude_gain": 0.998,    # Yêu cầu Fan Magnitude giảm và < 1
        "atr_threshold": 0.0015,
    }

    # --- Sell (Exit) Parameters ---
    sell_params = {
        "sell_trend_indicator": "trend_close_2h",  # EMA 2h làm chỉ báo thoát lệnh
    }

    plot_config = {
        'main_plot': {
            'senkou_a': {
                'color': 'green',
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud',
                'fill_color': 'rgba(255,76,46,0.2)',
            },
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'}
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            },
            'atr': {
                'atr': {'color': 'purple'}
            }
        }
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        # Sử dụng mức đòn bẩy cố định
        return 5#float(min(self.leverage_value, max_leverage))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Tạo các cột chỉ báo, trong đó:
          - Các cột Heikin Ashi (ha_open, ha_high, ha_low, ha_close) được thêm mới, không thay thế dữ liệu gốc.
          - Các chỉ báo EMA, Ichimoku và Fan Magnitude được tính toán từ dữ liệu đã làm mịn.
        """
        # Tạo dữ liệu Heikin Ashi riêng mà không ghi đè OHLC gốc
        ha = qtpylib.heikinashi(dataframe.copy())
        dataframe['ha_open'] = ha['open']
        dataframe['ha_high'] = ha['high']
        dataframe['ha_low'] = ha['low']
        dataframe['ha_close'] = ha['close']

        # Tính EMA dựa trên dữ liệu Heikin Ashi
        dataframe['trend_close_5m'] = dataframe['ha_close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['ha_close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['ha_close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['ha_close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['ha_close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['ha_close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['ha_close'], timeperiod=96)

        # EMA cho giá mở (dùng dữ liệu Heikin Ashi)
        dataframe['trend_open_5m'] = dataframe['ha_open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['ha_open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['ha_open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['ha_open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['ha_open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['ha_open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['ha_open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['ha_open'], timeperiod=96)

        # Tính Fan Magnitude (EMA 1h/EMA 8h) và sự thay đổi của nó
        dataframe['fan_magnitude'] = dataframe['trend_close_1h'] / dataframe['trend_close_8h']
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        # Tính Ichimoku Cloud (có thể tính dựa trên dữ liệu gốc hoặc HA tùy mục tiêu)
        ichimoku = ftt.ichimoku(
            dataframe.copy(),
            conversion_line_period=20,
            base_line_periods=60,
            laggin_span=120,
            displacement=30
        )
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        # Tính ATR từ dữ liệu gốc để hỗ trợ quản lý rủi ro
        dataframe['atr'] = ta.ATR(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Xác định tín hiệu vào lệnh cho cả long và short, kèm theo bộ lọc (ví dụ: ATR) để hạn chế giao dịch khi thị trường không có xu hướng.
        """
        # --- Tín hiệu LONG ---
        long_conditions = []
        level = self.buy_params['buy_trend_above_senkou_level']
        if level >= 1:
            long_conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            long_conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])
        if level >= 2:
            long_conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            long_conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])
        
        bullish_level = self.buy_params['buy_trend_bullish_level']
        if bullish_level >= 1:
            long_conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])
        if bullish_level >= 2:
            long_conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])
        if bullish_level >= 3:
            long_conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])
        
        long_conditions.append(dataframe['fan_magnitude'] > 1)
        long_conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain'])
        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            long_conditions.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])
        
        # Bộ lọc ATR để tránh vào lệnh khi biến động quá thấp
        long_conditions.append(dataframe['atr'] >= self.buy_params.get('atr_threshold', 0))
        
        # --- Tín hiệu SHORT ---
        short_conditions = []
        level_short = self.short_params['short_trend_below_senkou_level']
        if level_short >= 1:
            short_conditions.append(dataframe['trend_close_5m'] < dataframe['senkou_a'])
            short_conditions.append(dataframe['trend_close_5m'] < dataframe['senkou_b'])
        if level_short >= 2:
            short_conditions.append(dataframe['trend_close_15m'] < dataframe['senkou_a'])
            short_conditions.append(dataframe['trend_close_15m'] < dataframe['senkou_b'])
        
        bearish_level = self.short_params['short_trend_bearish_level']
        if bearish_level >= 1:
            short_conditions.append(dataframe['trend_close_5m'] < dataframe['trend_open_5m'])
        if bearish_level >= 2:
            short_conditions.append(dataframe['trend_close_15m'] < dataframe['trend_open_15m'])
        if bearish_level >= 3:
            short_conditions.append(dataframe['trend_close_30m'] < dataframe['trend_open_30m'])
        
        short_conditions.append(dataframe['fan_magnitude'] < 1)
        short_conditions.append(dataframe['fan_magnitude_gain'] <= self.short_params['short_max_fan_magnitude_gain'])
        for x in range(self.short_params['short_fan_magnitude_shift_value']):
            short_conditions.append(dataframe['fan_magnitude'].shift(x+1) > dataframe['fan_magnitude'])
        
        # Bộ lọc ATR cho tín hiệu short
        short_conditions.append(dataframe['atr'] >= self.short_params.get('atr_threshold', 0))
        
        if long_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, long_conditions), 'enter_long'] = 1
        if short_conditions:
            dataframe.loc[reduce(lambda x, y: x & y, short_conditions), 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Xác định tín hiệu thoát lệnh:
          - LONG: Khi giá 5m cắt xuống dưới EMA của khung 2h (trend_close_2h).
          - SHORT: Khi giá 5m cắt lên trên EMA của khung 2h.
        """
        exit_long_condition = qtpylib.crossed_below(dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']])
        dataframe.loc[exit_long_condition, 'exit_long'] = 1
        
        exit_short_condition = qtpylib.crossed_above(dataframe['trend_close_5m'], dataframe[self.sell_params['sell_trend_indicator']])
        dataframe.loc[exit_short_condition, 'exit_short'] = 1
        
        return dataframe
