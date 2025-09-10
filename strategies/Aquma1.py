# 파일명: user_data/strategies/OptimalStrategy.py
from freqtrade.strategy import IStrategy
import pandas as pd
import talib.abstract as ta

class OptimalStrategy(IStrategy):
    # 거래 시간 프레임 및 초기 캔들 수
    timeframe = '5m'
    startup_candle_count = 50

    # ROI (이익 실현) 설정: 거래 보유 기간에 따른 목표 이익률
    minimal_roi = {
        "0": 0.10,    # 즉시 10% 이익 시 실현
        "30": 0.05,   # 30분 이후 5% 이익 시 실현
        "60": 0.02,   # 60분 이후 2% 이익 시 실현
        "120": 0      # 2시간 이후에는 목표 없음(트레일링 스탑 활용)
    }

    # 고정 손절 설정: 10% 손실 시 종료
    stoploss = -0.10

    # 트레일링 스탑 설정: 이익 2% 도달 시 활성화, 0.5% 하락 시 청산
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Deprecated 옵션 변경: sell_profit_only -> exit_profit_only
    exit_profit_only = False

    # 추가 매도 신호 사용
    use_exit_signal = True
    ignore_roi_if_entry_signal = True

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        매매 신호 생성을 위한 기술적 지표 계산
        """
        # 단기, 장기 EMA 계산 (추세 확인)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=26)
        # ADX (추세 강도 측정)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        # RSI (모멘텀 및 과매수/과매도 판단)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # Bollinger Bands 하한선 (딥스 감지)
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        진입(매수) 신호 생성:
        - 상승 추세에서 가격이 일시적으로 하락(딥스)하거나 모멘텀이 강할 경우 진입
        """
        dataframe.loc[:, 'buy'] = 0

        # 상승 추세: 단기 EMA가 장기 EMA 위에 있으며, ADX가 25 이상인 경우
        uptrend = (dataframe['ema_fast'] > dataframe['ema_slow']) & (dataframe['adx'] > 25)
        # 딥스 조건: 가격이 Bollinger 하한선 이하이거나 RSI가 30 미만인 경우
        dip = (dataframe['close'] < dataframe['bb_lower']) | (dataframe['rsi'] < 30)
        # 모멘텀 돌파 조건: 단기 EMA > 장기 EMA 이면서 RSI가 70 이상인 경우
        momentum = (dataframe['ema_fast'] > dataframe['ema_slow']) & (dataframe['rsi'] > 70)

        entry_condition = (uptrend & dip) | momentum
        dataframe.loc[entry_condition, 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        청산(매도) 신호 생성:
        - 추세 반전, 과매수 상황 또는 가격 급락 시 청산
        """
        dataframe.loc[:, 'sell'] = 0

        # 추세 반전: 단기 EMA가 장기 EMA 아래로 교차하는 경우
        trend_reversal = dataframe['ema_fast'] < dataframe['ema_slow']
        # 과매수: RSI가 80 이상인 경우
        rsi_overbought = dataframe['rsi'] > 80
        # 가격 급락: 가격이 단기 EMA의 98% 이하로 떨어진 경우
        price_crash = dataframe['close'] < dataframe['ema_slow'] * 0.98

        exit_condition = trend_reversal | rsi_overbought | price_crash
        dataframe.loc[exit_condition, 'sell'] = 1

        return dataframe
