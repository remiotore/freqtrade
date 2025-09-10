from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from functools import reduce

class MenOfSteel(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False  # 숏 포지션을 사용하지 않음
    leverage_p = 1
    stoploss_p = - 0.03
    minstop = 0.003
    minimal_roi = {
        "0": 0.067,
        "8": 0.014,
        "20": 0.003,
        "30": 0
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.leverage_p = self.config.get('trade', {}).get('leverage', 1)
        self.stoploss_p = self.config.get('trade', {}).get('stoploss', -0.03) * self.leverage_p
        self.minstop = self.leverage_p * 0.003
        self.stoploss = self.stoploss_p
        self.full_leverage = float(self.config.get('trade', {}).get('leverage', 1))
        self.minimal_roi = self.generate_minimal_roi(self.full_leverage)
        #self.minimal_roi = self.leverage_roi_map.get(self.leverage_p, self.leverage_roi_map[1])
    exit_params = {
        # 0 5m / 1 15m / 2 30m / 3 1h / 4 2h / 5 4h / 6 8h
        "exit_trend_indicator": 4,
        #"exit_adx_threshold": 15   # ADX가 15 이하일 때 매도
    }
    def generate_minimal_roi(self, leverage: float):
        base_roi = {
            "0": 0.046,
            "6": 0.018,
            "15": 0.007,
            "20": 0.003,
        }

        return {k: round(v * self.leverage_p, 6) for k, v in base_roi.items()}
    # ?섏씠?쇳뙆?쇰????뺤쓽
    left = IntParameter(5, 60, default=12, space="buy")
    right = IntParameter(2, 30, default=8, space="buy")
    #maintain_candles = IntParameter(1, 10, default=9, space='buy', optimize=True)
    # 여기서 threshold 추가

    #position_amount = DecimalParameter(5, 50, decimals=1, default=10, space="stake")
    #position_amount = 10
    #minimal_roi_value = DecimalParameter(0.01, 0.2, decimals=3, default=0.052, space="sell")
    #stoploss_value = DecimalParameter(-0.99, -0.01, decimals=3, default=-0.084, space="sell")
    timeframe = '3m'

    trailing_stop = False
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        #return float(self.config.get('trade', {}).get('leverage', 1))
        #return self.leverage
        return self.full_leverage

    def f_pivothigh_confirmed(self, series: pd.Series, left: int, right: int) -> pd.Series:
        """
        - series: 고가(high)값 시리즈
        - left: 과거 몇 봉을 볼지
        - right: 미래 몇 봉이 닫힌 뒤 확정할지
        반환: pivot_highs 시리즈. 값이 찍히는 시점이 바로 '미래 right개봉이 다 닫힌 순간'인 i.
        """
        length = len(series)
        pivot_highs = [np.nan] * length

        # i가 최소 left+right 이상이어야, center = i-right ≥ left 보장
        for i in range(left + right, length):
            center = i - right

            # 과거 left개( center-left : center )
            window_left = series[center - left: center]
            # 미래 right개 ( center+1 : center+1+right )
            window_right = series[center + 1: center + 1 + right]

            center_val = series[center]

            # 과거 left 중 최고값, 미래 right 중 최고값 비교
            if (center_val >= window_left.max()) and (center_val > window_right.max()):
                # pivot 고점 확정은 “미래 right개봉까지 닫힌 시점”인 i 에 찍어준다.
                pivot_highs[i] = center_val

        return pd.Series(pivot_highs, index=series.index)

    def f_pivotlow_confirmed(self, series: pd.Series, left: int, right: int) -> pd.Series:
        """
        - series: 저가(low)값 시리즈
        - left, right 설명은 f_pivothigh_confirmed와 동일
        """
        length = len(series)
        pivot_lows = [np.nan] * length

        for i in range(left + right, length):
            center = i - right

            window_left = series[center - left: center]
            window_right = series[center + 1: center + 1 + right]
            center_val = series[center]

            # 과거 left 중 최저값, 미래 right 중 최저값 비교
            if (center_val <= window_left.min()) and (center_val < window_right.min()):
                # pivot 저점 확정은 “미래 right개봉까지 닫힌 시점”인 i 에 찍어준다.
                pivot_lows[i] = center_val

        return pd.Series(pivot_lows, index=series.index)
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pivotHigh = self.f_pivothigh_confirmed(dataframe['high'], self.left.value, self.right.value)
        pivotLow = self.f_pivotlow_confirmed(dataframe['low'], self.left.value, self.right.value)

        dataframe['pivotHigh'] = pivotHigh
        dataframe['pivotLow'] = pivotLow
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['buy'] = 0

        # pivotLow 이후 close가 상승 추세인지 확인
        dataframe['pivotLowValid'] = (
                (dataframe['pivotLow'].notnull()) &
                (dataframe['close'] > dataframe['close'].shift(1)) &
                (dataframe['close'].shift(1) > dataframe['close'].shift(2))
        )

        dataframe.loc[
            dataframe['pivotLowValid'],
            'buy'
        ] = 1

        return dataframe
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe['sell'] = 0
        dataframe.loc[
            (dataframe['pivotHigh'].notnull()),
            'sell'
        ] = 1

        return dataframe
    #def custom_entry_amount(self, pair: str, current_time, current_rate, current_profit, **kwargs):
    #    return self.position_amount.value

    def custom_exit(self, pair: str, current_time, current_rate, current_profit, **kwargs):
        # 손절 조건
        if current_profit <= self.stoploss:
            return True  # 손절 (즉시 매도)


        return False

