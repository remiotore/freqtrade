# -*- coding: utf-8 -*-
"""
Aquma6 Strategy (M8 변형):
- Smoothed Heikin-Ashi(SHA) + CDV(Cumulative Delta Volume) + STC(Schaff Trend Cycle)
- 롱: SHA 2연속 초록, CDV > 50SMA, STC >= 80  → BUY_M8 조건
- 숏: SHA 2연속 빨강, CDV < 50SMA, STC <= 20  → SELL_M8 조건
- 손절 및 목표수익은 config/stoploss, minimal_roi에서 설정하세요.
"""

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy

class Aquma7(IStrategy):
    INTERFACE_VERSION: int = 3
    can_short: bool = True
    timeframe = "1h"
    startup_candle_count = 60  # 충분한 지표 산출을 위한 최소 봉 수

    # ROI 및 Stoploss (예시)
    minimal_roi = {"0": 0.1, "30": 0.05, "60": 0.02, "120": 0.01}
    stoploss = -0.05

    # Trailing Stop (옵션)
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # 주문 매핑 (Futures 모드에서 unified 방식)
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC",
        "stoploss": "GTC",
        "stoploss_on_exchange": "GTC"
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        [1] Smoothed Heikin-Ashi (SHA)
        - 기본 HA 계산 후, 이중 EMA smoothing 적용.
        - sha_color: sha_close > sha_open 이면 "green", 아니면 "red".
        """
        o = dataframe['open'].values
        h = dataframe['high'].values
        l = dataframe['low'].values
        c = dataframe['close'].values
        n = len(dataframe)

        ha_close = np.zeros(n)
        ha_open  = np.zeros(n)
        ha_high  = np.zeros(n)
        ha_low   = np.zeros(n)

        # 초기값
        ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4
        ha_open[0] = (o[0] + c[0]) / 2
        ha_high[0] = max(h[0], ha_open[0], ha_close[0])
        ha_low[0] = min(l[0], ha_open[0], ha_close[0])

        for i in range(1, n):
            ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high[i] = max(h[i], ha_open[i], ha_close[i])
            ha_low[i] = min(l[i], ha_open[i], ha_close[i])

        dataframe['ha_open'] = ha_open
        dataframe['ha_close'] = ha_close
        dataframe['ha_high'] = ha_high
        dataframe['ha_low'] = ha_low

        # 이중 EMA smoothing
        def ema_np(arr, period):
            return pd.Series(arr).ewm(span=period, adjust=False).mean().values

        length_ha = 10
        length_ha2 = 10
        sha_open = ema_np(ema_np(ha_open, length_ha), length_ha2)
        sha_close = ema_np(ema_np(ha_close, length_ha), length_ha2)

        dataframe['sha_open'] = sha_open
        dataframe['sha_close'] = sha_close
        # SHA 색상 결정
        dataframe['sha_color'] = np.where(sha_close > sha_open, "green", "red")

        # [2] CDV (Cumulative Delta Volume)
        # 계산: (close - open) * volume 의 누적합, 그리고 50기간 SMA
        dataframe['cdv_delta'] = (dataframe['close'] - dataframe['open']) * dataframe['volume']
        dataframe['cdv'] = dataframe['cdv_delta'].cumsum()
        dataframe['cdv_sma50'] = dataframe['cdv'].rolling(50, min_periods=1).mean()

        # [3] STC (Schaff Trend Cycle)
        # MACD 기반: EMA12 - EMA26 → stoch-like 변환
        dataframe['ema12'] = dataframe['close'].ewm(span=12, adjust=False).mean()
        dataframe['ema26'] = dataframe['close'].ewm(span=26, adjust=False).mean()
        dataframe['macd_line'] = dataframe['ema12'] - dataframe['ema26']
        length_stc = 10
        dataframe['LL_macd'] = dataframe['macd_line'].rolling(length_stc, min_periods=1).min()
        dataframe['HH_macd'] = dataframe['macd_line'].rolling(length_stc, min_periods=1).max()
        df_range = dataframe['HH_macd'] - dataframe['LL_macd']
        dataframe['stc_raw'] = np.where(
            df_range > 1e-9,
            (dataframe['macd_line'] - dataframe['LL_macd']) / df_range * 100,
            np.nan
        )
        dataframe['stc'] = dataframe['stc_raw'].ffill()
        # STC 색상: >=80 => "green", <=20 => "red", 그 외 "none"
        dataframe['stc_color'] = dataframe['stc'].apply(lambda x: "green" if x >= 80 else ("red" if x <= 20 else "none"))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        진입 신호 설정:
         - 롱: 현재 및 이전 SHA 색상이 "green" AND cdv > cdv_sma50 AND stc_color=="green"
         - 숏: 현재 및 이전 SHA 색상이 "red" AND cdv < cdv_sma50 AND stc_color=="red"
        """
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # 이전 SHA 색상 (shift 1)
        df_sha_prev = dataframe['sha_color'].shift(1)

        # 롱 진입 조건
        long_condition = (
            (dataframe['sha_color'] == "green") &
            (df_sha_prev == "green") &
            (dataframe['cdv'] > dataframe['cdv_sma50']) &
            (dataframe['stc_color'] == "green")
        )
        dataframe.loc[long_condition, "enter_long"] = 1

        # 숏 진입 조건
        short_condition = (
            (dataframe['sha_color'] == "red") &
            (df_sha_prev == "red") &
            (dataframe['cdv'] < dataframe['cdv_sma50']) &
            (dataframe['stc_color'] == "red")
        )
        dataframe.loc[short_condition, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        청산 신호 설정:
         - 롱 청산: STC 색상이 "red"일 때 (stc가 하락 신호로 전환)
         - 숏 청산: STC 색상이 "green"일 때 (stc가 상승 신호로 전환)
         
        (실제 청산은 stoploss와 trailing_stop으로 관리되므로, exit 신호는 보조 용도로 사용)
        """
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        exit_long_condition = (dataframe['stc_color'] == "red")
        exit_short_condition = (dataframe['stc_color'] == "green")

        dataframe.loc[exit_long_condition, "exit_long"] = 1
        dataframe.loc[exit_short_condition, "exit_short"] = 1

        return dataframe
