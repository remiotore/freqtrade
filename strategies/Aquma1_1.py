import pandas as pd
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from typing import Optional

class Aquma1(IStrategy):
    # 기본 설정
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 1000
    can_short = True
    use_exit_signal = True

    stoploss = -0.02
    minimal_roi = {
        "0": 0.04,
        "60": 0.02,
        "120": 0.01,
        "240": 0
    }

    def __init__(self, config: dict, **kwargs):
        super().__init__(config, **kwargs)
        from xgboost import XGBClassifier
        import joblib
        self.model = None
        try:
            self.model = joblib.load('/freqtrade/user_data/models/aquma1_xgb.pkl')
        except Exception as e:
            print(f"모델 로드 실패: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # 기술적 지표 계산 (talib 사용)
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['MFI'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['ADX'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['EMA_20'] = ta.EMA(dataframe['close'], timeperiod=20)
        
        # MACD 계산
        macd, macdsignal, macdhist = ta.MACD(
            dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        dataframe['MACD'] = macd
        dataframe['MACD_signal'] = macdsignal
        dataframe['MACD_hist'] = macdhist
        
        # Bollinger Bands 계산 (nbdevup/nbdevdn는 float로 전달)
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
        )
        dataframe['BB_upper'] = bb_upper
        dataframe['BB_middle'] = bb_middle
        dataframe['BB_lower'] = bb_lower
        
        # Stochastic Oscillator 계산
        stoch_k, stoch_d = ta.STOCH(
            dataframe['high'], dataframe['low'], dataframe['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        dataframe['Stoch_%K'] = stoch_k
        dataframe['Stoch_%D'] = stoch_d
        
        # Directional Indicators 계산
        dataframe['DI_plus'] = ta.PLUS_DI(
            dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14
        )
        dataframe['DI_minus'] = ta.MINUS_DI(
            dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14
        )
        
        # 결측치 제거 대신 채우기를 사용하여 데이터프레임 길이 유지
        dataframe.fillna(0, inplace=True)
        
        # 모델 입력 피처 준비 (훈련시 사용한 순서와 동일하게)
        feature_columns = [
            'RSI', 'MFI', 'ADX', 'EMA_20', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'Stoch_%K', 'Stoch_%D', 'DI_plus', 'DI_minus'
        ]
        features = dataframe[feature_columns].values
        
        # 머신러닝 모델 예측 (상승 확률)
        if self.model:
            try:
                proba = self.model.predict_proba(features)
                dataframe['pred_prob'] = proba[:, 1]
            except Exception as e:
                dataframe['pred_prob'] = 0.0
        else:
            dataframe['pred_prob'] = 0.0

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0

        # 조건 완화 (예측 확률, RSI, EMA 및 ADX 조건)
        long_conditions = [
            dataframe['pred_prob'] > 0.55,
            dataframe['RSI'] < 40,
            dataframe['close'] > dataframe['EMA_20'],
            dataframe['ADX'] > 20
        ]
        if long_conditions:
            dataframe.loc[np.logical_and.reduce(long_conditions), 'enter_long'] = 1

        short_conditions = [
            dataframe['pred_prob'] < 0.45,
            dataframe['RSI'] > 60,
            dataframe['close'] < dataframe['EMA_20'],
            dataframe['ADX'] > 20
        ]
        if short_conditions:
            dataframe.loc[np.logical_and.reduce(short_conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0

        # 청산 조건 설정: 롱 포지션은 예측 확률이 낮거나 RSI가 높으면, 숏 포지션은 반대로 설정
        exit_long_cond = (dataframe['pred_prob'] < 0.5) | (dataframe['RSI'] > 60)
        exit_short_cond = (dataframe['pred_prob'] > 0.5) | (dataframe['RSI'] < 40)
        dataframe.loc[exit_long_cond, 'exit_long'] = 1
        dataframe.loc[exit_short_cond, 'exit_short'] = 1

        return dataframe
