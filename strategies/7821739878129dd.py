# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series, DatetimeIndex, merge
# --------------------------------
import talib.abstract as ta
import pandas_ta as pta
import numpy as np
import pandas as pd  # noqa
import warnings, datetime
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import stoploss_from_open, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from functools import reduce

pd.options.mode.chained_assignment = None  # default='warn'

# ------- 策略优化：Mastaaa1987 原版逻辑 + 动态参数与风控增强

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R 优化计算，避免除零错误"""
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    # 处理可能的零值（如价格长时间横盘）
    denominator = (highest_high - lowest_low).replace(0, 1e-6)
    WR = Series(
        (highest_high - dataframe["close"]) / denominator,
        name=f"{period} Williams %R",
    )
    return WR * -100

class OptimizedKamaFama_2(IStrategy):
    INTERFACE_VERSION = 2

    # --- 动态参数配置 ---
    buy_r_14 = DecimalParameter(-90, -40, default=-61.3, space='buy', optimize=True, load=True)
    buy_mama_diff = DecimalParameter(-0.05, 0, default=-0.025, space='buy', optimize=True, load=True)
    buy_cti = DecimalParameter(-1.0, -0.5, default=-0.715, space='buy', optimize=True, load=True)
    buy_volume_ratio = DecimalParameter(1.0, 2.0, default=1.5, decimals=1, space='buy', optimize=True, load=True)
    sell_fastx = IntParameter(70, 95, default=84, space='sell', optimize=True, load=True)
    
    # --- 风控参数 ---
    stoploss = -0.25  # 初始硬止损
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1% 回撤触发追踪止损
    trailing_stop_positive_offset = 0.03  # 盈利3%后激活

    # --- 时间框架与数据配置 ---
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 150  # 从400优化为150，加快启动

    # --- ROI 策略 ---
    minimal_roi = {
        "0": 0.15,   # 15% 立即止盈（适配高波动）
        "30": 0.10,  # 30分钟后降为10%
        "60": 0.05   # 60分钟后降为5%
    }

    # --- 订单配置 ---
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # --- 指标计算 ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 基础指标
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.25, 0.025)
        dataframe['mama_diff'] = (dataframe['mama'] - dataframe['fama']) / dataframe['hl2']
        dataframe['kama'] = ta.KAMA(dataframe['close'], 84)
        
        # 波动性指标（ATR）
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # 动量与超买超卖
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)
        
        # 成交量验证
        dataframe['volume_ma20'] = dataframe['volume'].rolling(20).mean()
        return dataframe

    # --- 入场逻辑（动态参数+量价确认）---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        buy_condition = (
            (dataframe['kama'] > dataframe['fama']) &
            (dataframe['fama'] > dataframe['mama'] * 0.981) &
            (dataframe['r_14'] < self.buy_r_14.value) &  # 参数化阈值
            (dataframe['mama_diff'] < self.buy_mama_diff.value) &
            (dataframe['cti'] < self.buy_cti.value) &
            (dataframe['volume'] > dataframe['volume_ma20'] * self.buy_volume_ratio.value) &  # 成交量放大
            (dataframe['close'] > dataframe['high'].shift(1)) &  # 突破前高
            (dataframe['rsi_84'] < 60) &
            (dataframe['rsi_112'] < 60)
        )
        conditions.append(buy_condition)
        dataframe.loc[buy_condition, 'enter_tag'] += 'dynamic_buy'

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1
        return dataframe

    # --- 动态止损（ATR波动性止损）---
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_atr = dataframe['atr'].iloc[-1]
        
        # 盈利超过5%时启用追踪止损
        if current_profit > 0.05:
            return -0.01  # 允许1%回撤
        
        # 基础止损：2倍ATR或硬止损取更优者
        stoploss_price = current_rate - 2 * current_atr
        hard_stoploss = current_rate * (1 + self.stoploss)  # 硬止损-25%
        final_stoploss = max(stoploss_price, hard_stoploss)
        return (final_stoploss / current_rate) - 1

    # --- 出场逻辑（优化FastK触发）---
    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        # 实时模式数据处理
        if self.config['runmode'].value in ('live', 'dry_run'):
            state = self.cc
            pc = state.get(trade.id, {
                'date': current_candle['date'],
                'high': current_candle['close'],
                'low': current_candle['close'],
                'close': current_rate
            })
            # 更新实时价格
            if current_rate > pc['high']:
                pc['high'] = current_rate
            if current_rate < pc['low']:
                pc['low'] = current_rate
            pc['close'] = current_rate
            state[trade.id] = pc

            # 重新计算FastK（避免延迟）
            df = dataframe.copy()
            df = df._append(pc, ignore_index=True)
            stoch_fast = ta.STOCHF(df, 5, 3, 0, 3, 0)
            fastk = stoch_fast['fastk'].iloc[-1]
        else:
            fastk = current_candle["fastk"]

        # 触发条件：FastK超阈值或最小持仓时间
        if fastk > self.sell_fastx.value:
            return "fastk_profit_sell"
        elif (current_time - trade.open_date_utc) > timedelta(hours=4) and current_profit > 0:
            return "time_based_exit"
        return None

    # --- 绘图配置 ---
    plot_config = {
        'main_plot': {
            "mama": {'color': '#d0da3e'},
            "fama": {'color': '#da3eb8'},
            "kama": {'color': '#3edad8'},
            "atr": {'color': '#808080', 'plotly': {'opacity': 0.3}}
        },
        "subplots": {
            "Volume": {
                "volume": {'color': '#4d4d4d'},
                "volume_ma20": {'color': '#ff9900'}
            },
            "Stoch": {
                "fastk": {'color': '#da3e3e'}
            }
        }
    }