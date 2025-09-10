import json
from pathlib import Path
from typing import Dict, Any
import freqtrade.vendor.qtpylib.indicators as qtpylib

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import IntParameter, DecimalParameter, RealParameter

class newstrategy_modify_support_fin(IStrategy):

    can_short = True
    timeframe = '5m'
    inf_timeframe = '1h'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # === 自动加载 JSON ===
        self.best_config: Dict[str, Any] = {}
        path = Path("user_data/strategies/newstrategy_modify_support_fin.json")
        if path.exists():
            try:
                with open(path) as f:
                    self.best_config = json.load(f).get("results", {}).get("best_config", {})
                print(f"[Hyperopt] Loaded config: {self.best_config}")
            except Exception as e:
                print(f"[Hyperopt] Load error: {e}")

        # === fallback 参数 ===
        self.fallback_buy_params = {
            "bbdelta_close": 0.00082, "bbdelta_tail": 0.85788, "close_bblower": 0.00128,
            "closedelta_close": 0.00987, "low_offset": 0.991, "rocr1_1h": 0.9346,
            "rocr_1h": 0.65666, "base_nb_candles_buy": 12, "buy_bb_width": 0.095,
            "buy_cci": -116, "buy_cci_length": 25, "buy_closedelta": 15.0,
            "buy_clucha_bbdelta_close": 0.049, "buy_clucha_bbdelta_tail": 1.146,
            "buy_clucha_close_bblower": 0.018, "buy_clucha_closedelta_close": 0.017,
            "buy_clucha_rocr_1h": 0.526, "buy_ema_diff": 0.025, "buy_rmi": 49,
            "buy_rmi_length": 17, "buy_roc_1h": 10, "buy_srsi_fk": 32,
            "ai_buy_threshold": 0.6
        }

        self.fallback_sell_params = {
            "high_offset": 1.012, "high_offset_2": 1.016, "sell_deadfish_bb_factor": 1.089,
            "sell_deadfish_bb_width": 0.11, "sell_deadfish_profit": -0.107,
            "sell_deadfish_volume_factor": 1.761, "base_nb_candles_sell": 22,
            "pHSL": -0.397, "pPF_1": 0.012, "pPF_2": 0.07, "pSL_1": 0.015,
            "pSL_2": 0.068, "sell_bbmiddle_close": 1.09092, "sell_fisher": 0.46406,
            "sell_trail_down_1": 0.03, "sell_trail_down_2": 0.015,
            "sell_trail_profit_max_1": 0.4, "sell_trail_profit_max_2": 0.11,
            "sell_trail_profit_min_1": 0.1, "sell_trail_profit_min_2": 0.04
        }

        def best(key: str, fallback: Any = None):
            return self.best_config.get(
                key,
                self.fallback_buy_params.get(key, self.fallback_sell_params.get(key, fallback))
            )
        # === BUY 参数 ===
        self.ai_buy_threshold = RealParameter(0.4, 0.9, default=best("ai_buy_threshold", 0.45), space='buy')
        self.rocr_1h = RealParameter(0.5, 1.0, default=best("rocr_1h", 0.65666), space='buy')
        self.rocr1_1h = RealParameter(0.5, 1.0, default=best("rocr1_1h", 0.9346), space='buy')
        self.bbdelta_close = RealParameter(0.0005, 0.02, default=best("bbdelta_close", 0.00082), space='buy')
        self.closedelta_close = RealParameter(0.0005, 0.02, default=best("closedelta_close", 0.00987), space='buy')
        self.bbdelta_tail = RealParameter(0.7, 1.2, default=best("bbdelta_tail", 0.85788), space='buy')
        self.close_bblower = RealParameter(0.0005, 0.02, default=best("close_bblower", 0.00128), space='buy')
        self.low_offset = DecimalParameter(0.985, 0.995, default=best("low_offset", 0.991), space='buy')
        self.base_nb_candles_buy = IntParameter(8, 20, default=best("base_nb_candles_buy", 12), space='buy')

        # === SELL 参数（示例）===
        self.sell_fisher = RealParameter(0.1, 0.5, default=best("sell_fisher", 0.46406), space='sell')
        self.sell_bbmiddle_close = RealParameter(0.97, 1.1, default=best("sell_bbmiddle_close", 1.09092), space='sell')
        self.high_offset = DecimalParameter(1.005, 1.015, default=best("high_offset", 1.012), space='sell')
        self.high_offset_2 = DecimalParameter(1.010, 1.020, default=best("high_offset_2", 1.016), space='sell')
        self.base_nb_candles_sell = IntParameter(8, 30, default=best("base_nb_candles_sell", 22), space='sell')

        # === 止损设置 ===
        self.stoploss = best("stoploss", -0.99)
        self.position_adjustment_enable = best("position_adjustment_enable", True)

        # === ROI 设置 ===
        self.minimal_roi = {
            "0": best("roi_0", 0.276),
            "32": best("roi_32", 0.105),
            "88": best("roi_88", 0.037),
            "208": best("roi_208", 0.0)
        }

        # === Trailing 止盈 ===
        self.trailing_stop = best("trailing_stop", False)
        self.trailing_stop_positive = best("trailing_stop_positive", 0.02)
        self.trailing_stop_positive_offset = best("trailing_stop_positive_offset", 0.10)
        self.trailing_only_offset_is_reached = best("trailing_only_offset_is_reached", True)
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if dataframe.empty:
            return dataframe

        # === Heikin Ashi candles ===
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']

        # === EWO: Elliott Wave Oscillator ===
        dataframe['ema_fast'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=200)
        dataframe['ewo'] = (dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['close'] * 100

        # === CTI: Correlation Trend Indicator ===
        def cti(series, length=20):
            diff = series.diff()
            up = diff.where(diff > 0, 0)
            down = -diff.where(diff < 0, 0)
            rs = up.rolling(length).mean() / down.rolling(length).mean()
            rsi = 100 - 100 / (1 + rs)
            return (rsi - 50) / 50

        dataframe['cti'] = cti(dataframe['close'], 20)

        # === Bollinger Bands ===
        upper, middle, lower = ta.BBANDS(dataframe['close'], timeperiod=20)
        
        dataframe['bb_upperband'] = upper
        dataframe['bb_middleband'] = middle
        dataframe['bb_lowerband'] = lower
        dataframe['bbdelta_close'] = abs(dataframe['bb_middleband'] - dataframe['close'])
        dataframe['closedelta_close'] = abs(dataframe['close'] - dataframe['close'].shift())
        dataframe['bbdelta_tail'] = abs(dataframe['close'] - dataframe['low'])
        dataframe['close_bblower'] = abs(dataframe['close'] - dataframe['bb_lowerband']) / dataframe['bb_lowerband']
        dataframe['bb_width'] = (upper - lower) / dataframe['close']

        # === 1h Informative Merge (for rocr_1h etc.) ===
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)
        informative['rocr_1h'] = ta.ROCR(informative['close'], timeperiod=28)
        informative['roc_1h'] = ta.ROC(informative['close'], timeperiod=9)
        upper, middle, lower = ta.BBANDS(informative['close'], timeperiod=20)
        informative['bb_upperband_1h'] = upper
        informative['bb_lowerband_1h'] = lower
        informative['bb_width_1h'] = (upper - lower) / informative['close']

        informative['rsi_1h'] = ta.RSI(informative['close'], timeperiod=14)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0

        # === 标签：FreqAI AI signal ===
        if 'lightgbm_prediction' in dataframe.columns:
            dataframe.loc[
                (dataframe['lightgbm_prediction'] > self.ai_buy_threshold.value),
                ['enter_long', 'enter_tag']
            ] = (1, 'freqai_signal')

        # === 标签：vwap 策略 ===
        dataframe.loc[
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['ewo'] < -5) &
            (dataframe['cti'] < -0.8),
            ['enter_long', 'enter_tag']
        ] = (1, 'vwap')

        # === 标签：cluc_HA 策略 ===
        dataframe.loc[
            (dataframe['ha_close'] < dataframe['bb_lowerband']) &
            (dataframe['bbdelta_close'] > self.bbdelta_close.value) &
            (dataframe['closedelta_close'] > self.closedelta_close.value) &
            (dataframe['bbdelta_tail'] < self.bbdelta_tail.value),
            ['enter_long', 'enter_tag']
        ] = (1, 'cluc_HA')

        # === 标签：insta_signal 策略 ===
        rsi = dataframe.get('rsi')
        if rsi is not None:
            dataframe.loc[
                (rsi < 56) &
                (dataframe['ewo'] > 5) &
                (dataframe['cti'] < -0.7),
            ['enter_long', 'enter_tag']
        ] = (1, 'insta_signal')

        # === 标签：NFINext7（示例）===
        rsi_1h = dataframe.get('rsi_1h')
        if rsi_1h is not None:
            dataframe.loc[
                (rsi_1h < -75) &
                (dataframe['ewo'] > 9.8) &
                (dataframe['cti'] < -0.8),
                ['enter_long', 'enter_tag']
            ] = (1, 'NFINext7')
        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0

        # === AI 辅助退出（FreqAI 反信号）===
        if 'lightgbm_prediction' in dataframe.columns:
            dataframe.loc[
                (dataframe['lightgbm_prediction'] < self.ai_buy_threshold.value * 0.9),
                ['exit_long', 'exit_tag']
            ] = (1, 'freqai_exit')

        # === 固定止盈逻辑（示例）===
        rsi = dataframe.get('rsi')
        if rsi is not None:
            dataframe.loc[
            (rsi > 70),
            ['exit_long', 'exit_tag']
        ] = (1, 'rsi>70')

        return dataframe

    # === 可选: 自定义 exit（用于 trailing 或其他条件）===
    # def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
    #                 current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
    #     # 你可以在这里接入 FreqAI 或技术指标作为 exit 逻辑
    #     return None

# === 策略类结束 ===
