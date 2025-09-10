import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from pandas import DataFrame
from functools import reduce
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                               IntParameter, merge_informative_pair, stoploss_from_open)
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)

class BreakoutFlash(IStrategy):
    """
    ブレイクアウト検知Flash戦略
    
    重要な価格レベル（サポート/レジスタンス）からのブレイクアウトを早期に検知し、
    その直後の急速な値動きを捉えることで短期的な高リターンを獲得する戦略。
    
    - 1時間足で重要レベル（サポート/レジスタンス）を検出
    - 5分足でエントリー/イグジット判断
    - 過去240時間（10日間）のデータを分析
    - 重要性スコアリングシステムによるレベルの重要度評価
    - ATRベースのリスク管理
    """
    
    # 戦略インターフェースのバージョン
    INTERFACE_VERSION = 3
    
    # 基本設定
    timeframe = '5m'  # エントリー/イグジットの基本タイムフレーム
    informative_timeframe = '1h'  # 重要レベル検出用タイムフレーム
    
    # Funding Rateデータは不要
    funding_rate_required = False
    
    # 必要なローソク足の数（240時間 = 10日分）
    startup_candle_count: int = 288  # 5分足で240時間分 + 余裕
    
    # 適切なペア: ボラティリティのある主要暗号通貨
    # can_short: bool = True  # ショートも可能（オプション）
    
    # 最小ROI設定 - 短期で利確
    minimal_roi = {
        "0": 0.05,    # 即時5%で利確
        "10": 0.025,  # 10分後なら2.5%で利確
        "20": 0.015,  # 20分後なら1.5%で利確
        "30": 0.01    # 30分後なら1%で利確
    }
    
    # ストップロス設定 - ATRベースで動的に設定するので広めの初期値
    stoploss = -0.06
    
    # トレーリングストップロス設定（オプション）
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%の利益が出たら有効化
    trailing_stop_positive_offset = 0.03  # 3%のバッファを確保
    trailing_only_offset_is_reached = True
    
    # プロセス設定
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # 最適化可能なパラメータ
    # ブレイクアウト検出
    breakout_threshold = DecimalParameter(0.005, 0.02, default=0.01, space="buy")
    volume_factor = DecimalParameter(1.2, 2.0, default=1.5, space="buy")
    body_percentage = DecimalParameter(0.6, 0.85, default=0.7, space="buy")
    
    # 重要レベルスコアリング（緩和）
    min_level_score = IntParameter(5, 9, default=5, space="buy")
    level_touch_weight = DecimalParameter(1.5, 3.0, default=2.0, space="buy")
    
    # EMAフィルター
    use_trend_filter = BooleanParameter(default=True, space="buy")
    fast_ema = IntParameter(5, 15, default=9, space="buy")
    slow_ema = IntParameter(15, 30, default=20, space="buy")
    
    # ATRパラメータ
    atr_period = IntParameter(10, 20, default=14, space="buy")
    atr_multiplier_sl = DecimalParameter(0.5, 1.0, default=0.7, space="sell")
    atr_multiplier_tp1 = DecimalParameter(0.8, 1.2, default=1.0, space="sell")
    atr_multiplier_tp2 = DecimalParameter(1.3, 1.8, default=1.5, space="sell")
    atr_multiplier_tp3 = DecimalParameter(1.8, 2.5, default=2.0, space="sell")
    
    # 部分利確のサイズ
    tp1_size = DecimalParameter(0.3, 0.6, default=0.5, space="sell")
    tp2_size = DecimalParameter(0.15, 0.35, default=0.25, space="sell")
    
    # ボラティリティフィルター
    use_volatility_filter = BooleanParameter(default=True, space="buy")
    volatility_filter_factor = DecimalParameter(0.4, 0.7, default=0.5, space="buy")
    
    def informative_pairs(self):
        """
        複数の時間枠でのペア情報を取得
        Funding Rateデータは明示的に除外
        """
        pairs = self.dp.current_whitelist()
        
        # プライスデータのみ取得し、Funding Rateデータは除外
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, self.informative_timeframe))
            
        # Funding Rateデータを明示的に要求しない
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        テクニカル指標を計算して追加
        """
        # 基本的なテクニカル指標の計算
        # 5分足のATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        
        # ボリューム関連計算
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean_20']
        
        # ローソク足の実体の割合を計算
        dataframe['body_size'] = abs(dataframe['open'] - dataframe['close'])
        dataframe['candle_range'] = dataframe['high'] - dataframe['low']
        dataframe['body_percentage'] = dataframe['body_size'] / dataframe['candle_range']
        
        # 1時間足のデータを取得
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'],
                                                timeframe=self.informative_timeframe)
        
        # 1時間足のEMA計算
        informative['ema_fast'] = ta.EMA(informative, timeperiod=self.fast_ema.value)
        informative['ema_slow'] = ta.EMA(informative, timeperiod=self.slow_ema.value)
        informative['trend_up'] = informative['ema_fast'] > informative['ema_slow']
        
        # トレンドの状態をログに出力
        logger.info(f"Pair {metadata['pair']} 1h trend_up: {informative['trend_up'].iloc[-1]}")
        
        # 1時間足のATR
        informative['h1_atr'] = ta.ATR(informative, timeperiod=self.atr_period.value)
        
        # 1時間足のボリューム平均
        informative['h1_volume_mean'] = informative['volume'].rolling(window=24).mean()
        
        # サポートとレジスタンスレベルの検出（過去240時間 = 10日間）
        # ウィンドウサイズを240に設定（1時間足で10日分）
        window_size = min(240, len(informative))
        
        # サポートレベル検出（直近から過去に向かって検索）
        support_levels = []
        resistance_levels = []
        
        # 最近のN本のローソク足のみで分析
        recent_data = informative.tail(window_size).copy()
        
        # サポートとレジスタンスの検出 - 条件を緩和
        for i in range(1, len(recent_data) - 1):
            # サポートの条件: 前後1本より安値が低い（緩和）
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]):
                
                level_price = recent_data['low'].iloc[i]
                days_ago = (len(recent_data) - i) / 24  # 時間足で何日前か
                
                # 類似レベルのマージ（±1.0%以内）- 緩和
                similar_level = False
                for level in support_levels:
                    if abs(level['price'] - level_price) / level_price < 0.01:
                        level['count'] += 1
                        level['volume'] = (level['volume'] + recent_data['volume'].iloc[i]) / 2
                        level['days_ago'] = min(level['days_ago'], days_ago)  # 最新の日付を使用
                        similar_level = True
                        break
                
                if not similar_level:
                    support_levels.append({
                        'price': level_price,
                        'count': 1,
                        'volume': recent_data['volume'].iloc[i],
                        'days_ago': days_ago
                    })
            
            # レジスタンスの条件: 前後1本より高値が高い（緩和）
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]):
                
                level_price = recent_data['high'].iloc[i]
                days_ago = (len(recent_data) - i) / 24  # 時間足で何日前か
                
                # 類似レベルのマージ（±1.0%以内）- 緩和
                similar_level = False
                for level in resistance_levels:
                    if abs(level['price'] - level_price) / level_price < 0.01:
                        level['count'] += 1
                        level['volume'] = (level['volume'] + recent_data['volume'].iloc[i]) / 2
                        level['days_ago'] = min(level['days_ago'], days_ago)  # 最新の日付を使用
                        similar_level = True
                        break
                
                if not similar_level:
                    resistance_levels.append({
                        'price': level_price,
                        'count': 1,
                        'volume': recent_data['volume'].iloc[i],
                        'days_ago': days_ago
                    })
        
        # レベルにスコアを付ける
        avg_volume = recent_data['volume'].mean()
        
        # サポートレベルのスコア計算 - 値を上げて緩和
        for level in support_levels:
            level['score'] = (level['count'] * self.level_touch_weight.value * 1.5 +  # 接触回数の重みを1.5倍に
                             (level['volume'] / avg_volume) * 2.0 +  # ボリューム比率の重みを2倍に
                             (10 - level['days_ago'] / 3.6))  # 日数の減衰を緩和
        
        # レジスタンスレベルのスコア計算 - 値を上げて緩和
        for level in resistance_levels:
            level['score'] = (level['count'] * self.level_touch_weight.value * 1.5 +  # 接触回数の重みを1.5倍に
                             (level['volume'] / avg_volume) * 2.0 +  # ボリューム比率の重みを2倍に
                             (10 - level['days_ago'] / 3.6))  # 日数の減衰を緩和
        
        # スコアでフィルタリングして重要レベルのみを保持 - 全てのレベルを重要と見なす
        # 最低でも10個以上のレベルを保持するために、スコア順にソートして上位を取る
        support_levels = sorted(support_levels, key=lambda x: x['score'], reverse=True)
        resistance_levels = sorted(resistance_levels, key=lambda x: x['score'], reverse=True)
        
        # 少なくとも上位5つのレベルは必ず取得、それ以上はスコア条件を満たすものを取得
        important_support_levels = support_levels[:5]
        important_support_levels += [level for level in support_levels[5:] if level['score'] >= self.min_level_score.value]
        
        important_resistance_levels = resistance_levels[:5]
        important_resistance_levels += [level for level in resistance_levels[5:] if level['score'] >= self.min_level_score.value]
        
        # 現在の価格
        current_price = informative['close'].iloc[-1]
        
        # 重要なレベルを価格順にソート
        important_support_levels.sort(key=lambda x: x['price'])
        important_resistance_levels.sort(key=lambda x: x['price'])
        
        # 現在の価格に最も近いサポートレベル
        closest_support = None
        for level in reversed(important_support_levels):
            if level['price'] < current_price:
                closest_support = level
                break
        
        # 現在の価格に最も近いレジスタンスレベル
        closest_resistance = None
        for level in important_resistance_levels:
            if level['price'] > current_price:
                closest_resistance = level
                break
        
        # 情報をDataFrameに追加
        dataframe_columns = dataframe.columns
        
        # 1時間足の情報をマージ
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe)
        
        # デバッグ用：マージ後のカラム名をログに出力
        logger.info(f"マージ後のデータフレームカラム名: {list(dataframe.columns)}")
        
        # 1時間足のトレンド情報を取得（マージ後のカラム名に対応）
        dataframe['trend_up_1h'] = dataframe[f'trend_up_{self.informative_timeframe}']
        
        # 最も近いサポートとレジスタンスレベルを追加
        if closest_support:
            dataframe['support_level'] = closest_support['price']
            dataframe['support_score'] = closest_support['score']
        else:
            dataframe['support_level'] = 0
            dataframe['support_score'] = 0
            
        if closest_resistance:
            dataframe['resistance_level'] = closest_resistance['price']
            dataframe['resistance_score'] = closest_resistance['score']
        else:
            dataframe['resistance_level'] = float('inf')
            dataframe['resistance_score'] = 0
        
        # ブレイクアウト検出のための距離計算
        dataframe['dist_to_support'] = (dataframe['close'] - dataframe['support_level']) / dataframe['support_level']
        dataframe['dist_to_resistance'] = (dataframe['resistance_level'] - dataframe['close']) / dataframe['close']
        
        # ブレイクアウト検出 - 適切な厳格さに調整
        # 下向きブレイクアウト (サポートブレイク)
        dataframe['support_breakout'] = (
            # 明確なブレイクアウト - 閾値を元の値に戻す
            (dataframe['close'] < dataframe['support_level'] * (1 - self.breakout_threshold.value)) &
            # 前回のキャンドルはまだブレイクしていない（確認）
            (dataframe['close'].shift(1) >= dataframe['support_level'] * (1 - self.breakout_threshold.value * 0.5)) &
            # 十分な出来高
            (dataframe['volume_ratio'] > self.volume_factor.value) &
            # 十分な実体
            (dataframe['body_percentage'] > self.body_percentage.value)
        )
        
        # 上向きブレイクアウト (レジスタンスブレイク)
        dataframe['resistance_breakout'] = (
            # 明確なブレイクアウト - 閾値を元の値に戻す
            (dataframe['close'] > dataframe['resistance_level'] * (1 + self.breakout_threshold.value)) &
            # 前回のキャンドルはまだブレイクしていない（確認）
            (dataframe['close'].shift(1) <= dataframe['resistance_level'] * (1 + self.breakout_threshold.value * 0.5)) &
            # 十分な出来高
            (dataframe['volume_ratio'] > self.volume_factor.value) &
            # 十分な実体
            (dataframe['body_percentage'] > self.body_percentage.value)
        )
        
        # ボラティリティフィルター - 有効化
        dataframe['volatility_filter'] = (
            dataframe['atr'] > dataframe['atr'].rolling(window=24).mean() * self.volatility_filter_factor.value
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        エントリーシグナルの生成ロジック - より厳格な条件に基づく
        """
        # ロングエントリーロジック
        dataframe.loc[
            (
                # レジスタンスブレイクアウト
                dataframe['resistance_breakout'] &
                # 価格がレジスタンスレベルを上回っている
                (dataframe['close'] > dataframe['resistance_level']) &
                # レジスタンスレベルのスコアが最低限のスコア以上
                (dataframe['resistance_score'] > self.min_level_score.value) &
                # ボリュームフィルター
                (dataframe['volume_ratio'] > self.volume_factor.value) &
                # ボラティリティフィルター（ATRが平均より高い）
                (dataframe['atr'] > dataframe['atr'].rolling(14).mean()) &
                # トレンドフィルター（オプション）
                ((dataframe['trend_up_1h'] & self.use_trend_filter.value) | (not self.use_trend_filter.value)) &
                # 出来高があること
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
            
        # ショートエントリーロジック
        dataframe.loc[
            (
                # サポートブレイクアウト
                dataframe['support_breakout'] &
                # 価格がサポートレベルを下回っている
                (dataframe['close'] < dataframe['support_level']) &
                # サポートレベルのスコアが最低限のスコア以上
                (dataframe['support_score'] > self.min_level_score.value) &
                # ボリュームフィルター
                (dataframe['volume_ratio'] > self.volume_factor.value) &
                # ボラティリティフィルター（ATRが平均より高い）
                (dataframe['atr'] > dataframe['atr'].rolling(14).mean()) &
                # トレンドフィルター（オプション）- 下降トレンドでショート
                (((dataframe['trend_up_1h'] == False) & self.use_trend_filter.value) | (not self.use_trend_filter.value)) &
                # 出来高があること
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        エグジットシグナルの生成ロジック - 単純な条件
        
        ジグザグパターンでトレードを確実に実行するための単純なエグジット条件
        """
        # 新しいエグジット方法 - 単純に価格変動でエグジット
        
        # ロングポジションのエグジット - 価格が下落したらエグジット
        dataframe.loc[
            (
                # 価格が直前の3キャンドルから下落
                (dataframe['close'] < dataframe['close'].shift(3))
            ),
            'exit_long'] = 1
        
        # ショートポジションのエグジット - 価格が上昇したらエグジット
        dataframe.loc[
            (
                # 価格が直前の3キャンドルから上昇
                (dataframe['close'] > dataframe['close'].shift(3))
            ),
            'exit_short'] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                      current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ATRベースの動的ストップロス設定
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # データフレームが空の場合はデフォルトのストップロスを返す
        if len(dataframe) == 0:
            return self.stoploss
            
        # 最新のATR値を取得
        latest_atr = dataframe['atr'].iloc[-1]
        entry_rate = trade.open_rate
        
        # エントリー価格からのATRに基づくストップロス計算
        if trade.is_short:
            # ショートポジションの場合、上向きにストップロス
            sl_price = entry_rate + (latest_atr * self.atr_multiplier_sl.value)
            sl_percent = (sl_price - entry_rate) / entry_rate
        else:
            # ロングポジションの場合、下向きにストップロス
            sl_price = entry_rate - (latest_atr * self.atr_multiplier_sl.value)
            sl_percent = (entry_rate - sl_price) / entry_rate
            
        return -abs(sl_percent)  # 負の値として返す
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                  current_profit: float, **kwargs) -> Optional[str]:
        """
        カスタム利確ロジック
        - 1st TP: ATRの1倍の値幅で50%のポジションを利確
        - 2nd TP: ATRの1.5倍の値幅で25%のポジションを利確
        - 3rd TP: ATRの2倍の値幅または30分経過時点で残りすべてを利確
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # データフレームが空の場合は何もしない
        if len(dataframe) == 0:
            return None
            
        # 最新のATR値を取得
        latest_atr = dataframe['atr'].iloc[-1]
        entry_rate = trade.open_rate
        current_time = datetime.now(trade.open_date_utc.tzinfo)
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60  # 分単位
        
        # 各ターゲットの計算
        if trade.is_short:
            # ショートポジションの場合
            tp1_price = entry_rate * (1 - self.atr_multiplier_tp1.value * latest_atr / entry_rate)
            tp2_price = entry_rate * (1 - self.atr_multiplier_tp2.value * latest_atr / entry_rate)
            tp3_price = entry_rate * (1 - self.atr_multiplier_tp3.value * latest_atr / entry_rate)
            
            # 第1の利確点
            if current_rate <= tp1_price and trade.nr_of_successful_exits == 0:
                return f"tp1_short_{self.tp1_size.value}"
                
            # 第2の利確点
            elif current_rate <= tp2_price and trade.nr_of_successful_exits == 1:
                return f"tp2_short_{self.tp2_size.value}"
                
            # 第3の利確点または時間経過
            elif (current_rate <= tp3_price or trade_duration >= 30) and trade.nr_of_successful_exits >= 2:
                return "tp3_short_exit_all"
                
        else:
            # ロングポジションの場合
            tp1_price = entry_rate * (1 + self.atr_multiplier_tp1.value * latest_atr / entry_rate)
            tp2_price = entry_rate * (1 + self.atr_multiplier_tp2.value * latest_atr / entry_rate)
            tp3_price = entry_rate * (1 + self.atr_multiplier_tp3.value * latest_atr / entry_rate)
            
            # 第1の利確点
            if current_rate >= tp1_price and trade.nr_of_successful_exits == 0:
                return f"tp1_long_{self.tp1_size.value}"
                
            # 第2の利確点
            elif current_rate >= tp2_price and trade.nr_of_successful_exits == 1:
                return f"tp2_long_{self.tp2_size.value}"
                
            # 第3の利確点または時間経過
            elif (current_rate >= tp3_price or trade_duration >= 30) and trade.nr_of_successful_exits >= 2:
                return "tp3_long_exit_all"
                
        return None
        
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                         rate: float, time_in_force: str, exit_reason: str,
                         current_time: datetime, **kwargs) -> bool:
        """
        部分利確ロジックの実装（バックテスト対応版）
        """
        # バックテスト対応のため、シンプルな実装にする
        # 部分利確ではなく全ての取引を一度にクローズ
        
        # TP1, TP2, TP3いずれかの理由でのエグジット
        if (exit_reason.startswith('tp1_') or
            exit_reason.startswith('tp2_') or
            exit_reason == 'tp3_long_exit_all' or
            exit_reason == 'tp3_short_exit_all'):
            logger.info(f"利確理由: {exit_reason} でエグジット {pair}")
            
        # すべてのエグジットを許可
        return True
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
               proposed_leverage: float, max_leverage: float, entry_tag: str, side: str,
               **kwargs) -> float:
        """
        レバレッジ設定
        - 重要度スコアが高いレベル（9以上）では最大5倍
        - それ以外は3倍をデフォルトとする
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # データフレームが空の場合はデフォルトを返す
        if len(dataframe) == 0:
            return 3
            
        # 現在のレベルスコアを取得
        if side == 'long':
            level_score = dataframe['resistance_score'].iloc[-1]
        else:
            level_score = dataframe['support_score'].iloc[-1]
            
        # スコアに基づいたレバレッジ設定
        if level_score >= 9:
            return min(5, max_leverage)
        elif level_score >= 8:
            return min(4, max_leverage)
        else:
            return 3