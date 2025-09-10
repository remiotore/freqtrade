import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional

import talib.abstract as ta
import pandas_ta as pta
from scipy.signal import argrelextrema

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)


class NOTankAi15(IStrategy):
    """
    Улучшенная стратегия на 15-минутном таймфрейме.

    Основные улучшения:
      - Динамический стоплосс на основе ATR.
      - Динамический расчёт кредитного плеча.
      - Расчёт уровней Murrey Math.
      - Расширенное логирование и структурирование кода.
    """

    # Общие параметры стратегии
    timeframe = "15m"
    startup_candle_count: int = 200
    stoploss = -0.99  # Базовый стоплосс, переопределяется в custom_stoploss
    trailing_stop = False
    position_adjustment_enable = True
    can_short = False
    use_exit_signal = True
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 2
    max_dca_multiplier = 1
    process_only_new_candles = True

    # Параметры DCA (улучшены для гибкости)
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 6, default=2, space="buy", optimize=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )

    # Параметры входа
    increment = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price = None

    # Параметры защиты
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # ROI: время – процент
    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }

    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            "min_max": {
                "maxima": {"color": "#a29db9", "type": "line"},
                "minima": {"color": "#ac7fc", "type": "line"},
                "maxima_check": {"color": "#a29db9", "type": "line"},
                "minima_check": {"color": "#ac7fc", "type": "line"},
            },
        },
    }

    @property
    def protections(self):
        """Методы защиты позиции."""
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,  # 3 дня (24 * 3)
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Корректирует объём входа для DCA."""
        return proposed_stake / self.max_dca_multiplier

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Расчитывает цену входа с учетом предыдущих входов и небольшой инкрементации,
        чтобы избежать идентичных значений.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        # Среднее из закрытия, открытия последней свечи и предложенной цены
        entry_price = (dataframe["close"].iat[-1] + dataframe["open"].iat[-1] + proposed_rate) / 3
        if proposed_rate < entry_price:
            entry_price = proposed_rate

        logger.info(f"{pair} Entry Price: {entry_price} | Close: {dataframe['close'].iat[-1]}, "
                    f"Open: {dataframe['open'].iat[-1]}, Proposed: {proposed_rate}")

        # Если разница меньше порогового значения, инкрементируем цену
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.0005:
            entry_price *= self.increment.value
            logger.info(f"{pair} Incremented entry price to {entry_price} (prev: {self.last_entry_price}).")
        self.last_entry_price = entry_price
        return entry_price

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Динамический стоплосс на основе ATR.
        Стоплосс рассчитывается как -1.5 * ATR, нормированный по цене.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        atr = ta.ATR(dataframe, timeperiod=14).iat[-1]
        # Пример: стоплосс = - (1.5 * ATR / текущая цена)
        dynamic_sl = -1.5 * atr / current_rate
        # Логируем рассчитанный стоплосс
        logger.info(f"{pair} Dynamic Stoploss: {dynamic_sl} (ATR: {atr}, Current Rate: {current_rate})")
        return dynamic_sl

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float,
                             time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """
        Подтверждение выхода из сделки.
        Если причина выхода связана с отрицательной прибылью, сделка не закрывается.
        """
        if exit_reason in ["partial_exit", "trailing_stop_loss"] and trade.calc_profit_ratio(rate) < 0:
            logger.info(f"{pair} Exit signal '{exit_reason}' отклонён, прибыль ниже 0.")
            self.dp.send_msg(f"{pair} Exit signal '{exit_reason}' отклонён, прибыль ниже 0.")
            return False
        return True

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float, current_entry_profit: float,
                              current_exit_profit: float, **kwargs) -> Optional[float]:
        """
        Корректировка позиции на основе текущей прибыли и числа входов/выходов.
        Если прибыль достаточно высока – часть позиции закрывается.
        """
        count_of_entries = trade.nr_of_successful_entries

        if current_profit > 0.25 and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount / 4)
        if current_profit > 0.40 and trade.nr_of_successful_exits == 1:
            return -(trade.stake_amount / 3)

        # Если убыток невелик, корректировка не требуется
        if (current_profit > -0.15 and count_of_entries == 1) or \
           (current_profit > -0.3 and count_of_entries == 2) or \
           (current_profit > -0.6 and count_of_entries == 3):
            return None

        try:
            stake_amount = trade.select_filled_orders(trade.entry_side)[0].cost
            return stake_amount
        except Exception as e:
            logger.error(f"Error adjusting trade position: {e}")
            return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        """
        Динамический расчёт кредитного плеча с использованием индикаторов RSI, ATR, MACD и SMA.
        """
        window_size = 50
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        close_prices = dataframe["close"].tail(window_size)
        high_prices = dataframe["high"].tail(window_size)
        low_prices = dataframe["low"].tail(window_size)
        base_leverage = 10

        rsi = ta.RSI(close_prices, timeperiod=14)
        atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        macd, macdsignal, _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        sma = ta.SMA(close_prices, timeperiod=20)

        current_rsi = rsi[-1] if len(rsi) > 0 else 50.0
        current_atr = atr[-1] if len(atr) > 0 else 0.0
        current_macd = (macd[-1] - macdsignal[-1]) if len(macd) > 0 and len(macdsignal) > 0 else 0.0
        current_sma = sma[-1] if len(sma) > 0 else current_rate

        # Пороговые значения для RSI
        dynamic_rsi_low = np.nanmin(rsi) if len(rsi) > 0 and not np.isnan(np.nanmin(rsi)) else 30.0
        dynamic_rsi_high = np.nanmax(rsi) if len(rsi) > 0 and not np.isnan(np.nanmax(rsi)) else 70.0

        # Факторы изменения кредитного плеча
        long_increase = 1.5
        long_decrease = 0.5
        volatility_decrease = 0.8

        if side == "long":
            if current_rsi < dynamic_rsi_low:
                base_leverage *= long_increase
            elif current_rsi > dynamic_rsi_high:
                base_leverage *= long_decrease

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease

            if current_macd > 0:
                base_leverage *= long_increase
            if current_rate < current_sma:
                base_leverage *= long_decrease

        adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)
        return adjusted_leverage

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Добавляет базовые индикаторы и уровни Murrey Math.
        """
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["DI_values"] = ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe)
        dataframe["DI_cutoff"] = 0

        # Определяем экстремумы (максимумы и минимумы)
        maxima = np.zeros(len(dataframe))
        minima = np.zeros(len(dataframe))
        maxima[argrelextrema(dataframe["close"].values, np.greater, order=5)] = 1
        minima[argrelextrema(dataframe["close"].values, np.less, order=5)] = 1
        dataframe["maxima"] = maxima
        dataframe["minima"] = minima

        # Экстремальные сигналы
        dataframe["s_extrema"] = 0
        min_peaks = argrelextrema(dataframe["close"].values, np.less, order=5)[0]
        max_peaks = argrelextrema(dataframe["close"].values, np.greater, order=5)[0]
        dataframe.loc[min_peaks, "s_extrema"] = -1
        dataframe.loc[max_peaks, "s_extrema"] = 1

        # Рассчитываем уровни Murrey Math
        murrey_levels = calculate_murrey_math_levels(dataframe)
        for level, series in murrey_levels.items():
            dataframe[level] = series

        # Дополнительный осциллятор (MML Extreme Oscillator)
        dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - dataframe["[4/8]P"]) /
                                                    (dataframe["[+3/8]P"] - dataframe["[-3/8]P"]))
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=10).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=10).max()
        dataframe["minima_check"] = dataframe["minima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        dataframe["maxima_check"] = dataframe["maxima"].rolling(4).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)

        pair = metadata.get("pair", "PAIR")
        if dataframe["maxima"].iloc[-3] == 1 and dataframe["maxima_check"].iloc[-1] == 0:
            self.dp.send_msg(f"*** {pair} *** Maxima Detected - Potential Short!!!")
        if dataframe["minima"].iloc[-3] == 1 and dataframe["minima_check"].iloc[-1] == 0:
            self.dp.send_msg(f"*** {pair} *** Minima Detected - Potential Long!!!")

        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Генерация сигналов для входа в позицию.
        """
        # Сигналы для длинных позиций
        df.loc[
            (df["DI_catch"] == 1) &
            (df["maxima_check"] == 1) &
            (df["s_extrema"] < 0) &
            (df["minima"].shift(1) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 30),
            ["enter_long", "enter_tag"]
        ] = (1, "Minima")

        df.loc[
            (df["minima_check"] == 0) &
            (df["volume"] > 0) &
            (df["rsi"] < 30),
            ["enter_long", "enter_tag"]
        ] = (1, "Minima Full Send")

        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 0) &
            (df["minima_check"].shift(5) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 30),
            ["enter_long", "enter_tag"]
        ] = (1, "Minima Check")

        # Сигналы для коротких позиций (если будут реализованы)
        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 1) &
            (df["s_extrema"] > 0) &
            (df["maxima"].shift(1) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] > 70),
            ["enter_short", "enter_tag"]
        ] = (1, "Maxima")

        df.loc[
            (df["maxima_check"] == 0) &
            (df["volume"] > 0) &
            (df["rsi"] > 70),
            ["enter_short", "enter_tag"]
        ] = (1, "Maxima Full Send")

        df.loc[
            (df["DI_catch"] == 1) &
            (df["maxima_check"] == 0) &
            (df["maxima_check"].shift(5) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] > 70),
            ["enter_short", "enter_tag"]
        ] = (1, "Maxima Check")

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Генерация сигналов для выхода из позиции.
        """
        df.loc[
            (df["maxima_check"] == 0) & (df["volume"] > 0),
            ["exit_long", "exit_tag"]
        ] = (1, "Maxima Check")
        df.loc[
            (df["DI_catch"] == 1) &
            (df["s_extrema"] > 0) &
            (df["maxima"].shift(1) == 1) &
            (df["volume"] > 0),
            ["exit_long", "exit_tag"]
        ] = (1, "Maxima")
        df.loc[
            (df["maxima_check"] == 0) & (df["volume"] > 0),
            ["exit_long", "exit_tag"]
        ] = (1, "Maxima Full Send")

        df.loc[
            (df["minima_check"] == 0) & (df["volume"] > 0),
            ["exit_short", "exit_tag"]
        ] = (1, "Minima Check")
        df.loc[
            (df["DI_catch"] == 1) &
            (df["s_extrema"] < 0) &
            (df["minima"].shift(1) == 1) &
            (df["volume"] > 0),
            ["exit_short", "exit_tag"]
        ] = (1, "Minima")
        df.loc[
            (df["minima_check"] == 0) & (df["volume"] > 0),
            ["exit_short", "exit_tag"]
        ] = (1, "Minima Full Send")
        return df


def calculate_murrey_math_levels(df: pd.DataFrame, window_size: int = 64) -> dict:
    """
    Рассчитывает уровни Murrey Math для каждого индекса в DataFrame.
    Возвращает словарь, где каждому уровню соответствует Series.
    """
    rolling_max_H = df["high"].rolling(window=window_size).max()
    rolling_min_L = df["low"].rolling(window=window_size).min()

    # Инициализируем словарь для уровней
    murrey_levels = {key: [] for key in ["[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
                                         "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
                                         "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"]}

    def calculate_mml(mn: float, finalH: float, mx: float, finalL: float):
        """
        Рассчитывает значение mml и возвращает уровни Murrey Math.
        """
        dmml = ((finalH - finalL) / 8) * 1.0699
        mml = (mx * 0.99875) + (dmml * 3)
        ml = [mml - (dmml * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0],
        }

    for i in range(len(df)):
        mn = df["low"].iloc[:i+1].min()
        mx = df["high"].iloc[:i+1].max()
        finalH = df["high"].iloc[:i+1].max()
        # Делим диапазон на 8 частей
        dmml = (mx - mn) / 8
        x_values = [mn + i * dmml for i in range(8)]
        midpoints = [(x_values[j] + x_values[j+1]) / 2 for j in range(7)]
        finalL = min(midpoints) if midpoints else mn

        levels = calculate_mml(mn, finalH, mx, finalL)
        for key in murrey_levels.keys():
            murrey_levels[key].append(levels.get(key, np.nan))

    # Преобразуем списки в Series с индексами DataFrame
    for key in murrey_levels:
        murrey_levels[key] = pd.Series(murrey_levels[key], index=df.index)
    return murrey_levels