import logging
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from typing import Dict, Optional
from pandas import DataFrame
import talib.abstract as ta
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
import requests
import time

class DCAbyGrok3Adapt_100_new(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = {
        "360": 0.004,
        "120": 0.012,
        "60": 0.008,
        "30": 0.006,
        "0": 0.004
    }

    stoploss = -0.08  # Помірний профіль: менш агресивний стоп-лосс
    trailing_stop = True
    trailing_stop_positive = 0.004  # Помірний профіль: зменшено
    trailing_stop_positive_offset = 0.012  # Помірний профіль: зменшено
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    max_open_trades = 2  # Помірний профіль: зменшено кількість відкритих угод
    position_adjustment_enable = True
    max_entry_position_adjustment = 2  # Помірний профіль: зменшено кількість DCA входів
    max_dca_multiplier = 0.4  # Помірний профіль: зменшено множник DCA
    rsi_upper_threshold = 75
    rsi_lower_threshold = 25

    buy_rsi = IntParameter(20, 40, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(60, 80, default=75, space="sell", optimize=True)
    risk_factor = DecimalParameter(0.3, 0.7, default=0.5, decimals=2, space="buy", optimize=True)
    leverage_factor = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="buy", optimize=True)  # Додано для кредитного плеча

    startup_candle_count = 200

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    logger = logging.getLogger(__name__)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.last_dca_level = {}
        self.news_api_key = config.get("news_api_key", "")
        self.initial_balance = 95.0  # Початковий баланс 95$
        if not self.news_api_key:
            self.logger.warning("NewsAPI key not provided in config.json. News filtering will be disabled.")

    def fetch_news(self, asset: str, current_time: datetime) -> bool:
        if not self.news_api_key:
            self.logger.debug(f"No NewsAPI key. Skipping news check for {asset}.")
            return False
        try:
            from_time = (current_time - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={asset}+crypto&from={from_time}&sortBy=publishedAt&apiKey={self.news_api_key}"
            )
            time.sleep(2)  # Затримка для NewsAPI
            response = requests.get(url, timeout=10)
            if response.status_code == 429:
                self.logger.warning(f"NewsAPI rate limit reached for {asset}. Skipping news check.")
                return False
            response.raise_for_status()
            news_data = response.json()
            if news_data.get("status") == "ok" and news_data.get("totalResults", 0) > 0:
                self.logger.info(f"Found {news_data['totalResults']} recent news articles for {asset} via NewsAPI. Blocking entry.")
                return True
            return False
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"NewsAPI request failed for {asset}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error fetching news for {asset}: {str(e)}", exc_info=True)
            return False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["sma_short"] = ta.SMA(dataframe, timeperiod=5)
        dataframe["sma_long"] = ta.SMA(dataframe, timeperiod=10)
        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=5)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=10)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe["stoch_rsi"] = stochrsi["fastk"]
        dataframe["min_price_20"] = dataframe["low"].rolling(window=20).min()
        dataframe["rsi_prev"] = dataframe["rsi"].shift(1)
        dataframe["close_prev"] = dataframe["close"].shift(1)
        dataframe["max_price_10"] = dataframe["high"].rolling(window=10).max()
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_sma_50"] = ta.SMA(dataframe["volume"], timeperiod=50)
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_lower"] = bollinger["lowerband"]
        dataframe["bb_middle"] = bollinger["middleband"]
        dataframe["bb_upper"] = bollinger["upperband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]
        # Додано VWAP
        dataframe["vwap"] = ((dataframe["close"] * dataframe["volume"]).cumsum() / dataframe["volume"].cumsum())
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        open_trades = Trade.get_open_trades()
        if any(trade.pair == pair for trade in open_trades):
            return dataframe

        last_candle = dataframe.iloc[-1]
        current_time = last_candle["date"].to_pydatetime()
        base_asset = pair.split("/")[0]
        has_recent_news = self.fetch_news(base_asset, current_time)
        if has_recent_news:
            self.logger.info(f"Blocking entry for {pair} due to recent news.")
            return dataframe

        dynamic_rsi = max(20, self.buy_rsi.value - (last_candle["atr"] / last_candle["close"] * 10))
        alternative_entry = (
            (dataframe["ema_short"] > dataframe["ema_long"]) &
            (dataframe["macd"] > dataframe["macd_signal"])
        )
        initial_entry = (
            (dataframe["rsi"] < dynamic_rsi) &
            (dataframe["sma_short"] > dataframe["sma_long"]) &
            (dataframe["adx"] > 25) &
            (dataframe["volume"] > dataframe["volume_sma"] * (2.0 * self.risk_factor.value)) &
            (dataframe["volume"] > dataframe["volume_sma_50"]) &
            (dataframe["stoch_rsi"] < 0.2) &
            (dataframe["close"] <= dataframe["bb_lower"] * 1.01) &
            (dataframe["bb_width"] < 0.05) &
            (dataframe["close"] > dataframe["vwap"] * 0.995)  # VWAP: ціна не нижче 99.5% від VWAP
        ) | alternative_entry

        dataframe.loc[initial_entry, "enter_long"] = 1
        dataframe["enter_long"] = dataframe["enter_long"].fillna(0).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        open_trades = Trade.get_open_trades()
        trade = next((t for t in open_trades if t.pair == pair), None)
        if not trade:
            return dataframe

        last_candle = dataframe.iloc[-1]
        current_profit = trade.calc_profit_ratio(last_candle["close"])
        volatility_factor = last_candle["atr"] / last_candle["close"]

        base_rsi = self.sell_rsi.value
        adaptive_rsi = min(95, base_rsi + (0.02 / (volatility_factor + 0.01) * 100))
        if last_candle["adx"] > 25:
            adaptive_rsi = min(95, adaptive_rsi + 5 * self.risk_factor.value)
        if current_profit > 0.01 * self.risk_factor.value:
            adaptive_rsi = min(95, adaptive_rsi + 3 * self.risk_factor.value)

        exit_condition = (
            (current_profit > 0.005 * self.risk_factor.value) &
            (
                (dataframe["rsi"] > adaptive_rsi) &
                (dataframe["rsi"] < dataframe["rsi_prev"]) &
                (dataframe["close"] < dataframe["max_price_10"] * 0.995) &
                (dataframe["close"] < dataframe["vwap"] * 1.005)  # VWAP: вихід, якщо ціна вище VWAP на 0.5%
            ) |
            (
                (dataframe["macd"] < dataframe["macd_signal"]) &
                (dataframe["adx"] < 20)
            )
        )

        dataframe.loc[exit_condition, "exit_long"] = 1
        dataframe["exit_long"] = dataframe["exit_long"].fillna(0).astype(int)
        return dataframe

    def custom_stake_amount(self, pair: str, current_rate: float, current_time: datetime, 
                           proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        available_balance = self.wallets.get_free("USDT")
        stake_percentage = 0.05 * self.risk_factor.value  # 5% від балансу, скоригованого на ризик
        fixed_stake = available_balance * stake_percentage * self.leverage_factor.value
        min_stake = max(min_stake, 1.0)  # Зменшено мінімальний стейк для малого балансу
        if fixed_stake < min_stake:
            self.logger.warning(f"Entry stake {fixed_stake} for {pair} below min_stake {min_stake}. Adjusting.")
            return min_stake
        if fixed_stake > max_stake:
            self.logger.warning(f"Entry stake {fixed_stake} for {pair} exceeds max_stake {max_stake}. Adjusting.")
            return max_stake
        if fixed_stake > available_balance:
            self.logger.warning(f"Entry stake {fixed_stake} for {pair} exceeds available balance {available_balance}. Adjusting.")
            return available_balance
        self.logger.info(f"Entry stake for {pair}: {fixed_stake:.2f} USDT (leverage: {self.leverage_factor.value}x)")
        return fixed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              **kwargs) -> Optional[float]:
        if not trade.is_open:
            return None

        pair = trade.pair
        orders_count = trade.nr_of_successful_entries
        if orders_count > self.max_entry_position_adjustment:
            self.logger.info(f"Max DCA entries ({self.max_entry_position_adjustment}) reached for {pair}. Skipping DCA.")
            return None

        avg_price = trade.open_rate
        close = current_rate
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if current_profit > 0.008 * self.risk_factor.value:
            amount_to_sell = trade.amount * 0.4 * self.risk_factor.value
            self.logger.info(f"Partial exit for {pair}: Selling {amount_to_sell:.6f} at {close:.6f}")
            return -amount_to_sell

        price_drop = (avg_price - close) / avg_price
        dca_step = 1.5 * last_candle["atr"] / last_candle["close"]
        max_price_drop = 3 * last_candle["atr"] / last_candle["close"]
        adx = last_candle["adx"]
        macd = last_candle["macd"]
        macd_signal = last_candle["macd_signal"]
        rsi = last_candle["rsi"]

        if adx < 15:
            self.logger.info(f"ADX ({adx:.2f}) too low for {pair}. Skipping DCA to avoid weak trend.")
            return None

        if macd < macd_signal and rsi > self.rsi_lower_threshold:
            self.logger.info(f"MACD ({macd:.6f}) below signal, but RSI ({rsi:.2f}) suggests recovery for {pair}. Proceeding with DCA.")
        elif macd >= macd_signal:
            pass
        else:
            self.logger.info(f"MACD ({macd:.6f}) below signal ({macd_signal:.6f}) for {pair}. Skipping DCA.")
            return None

        if price_drop >= max_price_drop:
            self.logger.info(f"Price drop for {pair} exceeds dynamic threshold ({price_drop:.2%}). Skipping DCA.")
            return None

        if trade.id not in self.last_dca_level:
            self.last_dca_level[trade.id] = 0.0

        current_dca_level = (price_drop // dca_step) * dca_step
        last_dca_level = self.last_dca_level[trade.id]

        if price_drop >= dca_step and current_dca_level > last_dca_level:
            self.last_dca_level[trade.id] = current_dca_level
            atr = last_candle["atr"]
            volatility_factor = atr / last_candle["close"]
            total_position = sum(o.stake_amount for o in trade.orders)

            if rsi < self.rsi_lower_threshold:
                new_stake = trade.stake_amount * 0.7 * self.risk_factor.value * (1 + volatility_factor) * self.leverage_factor.value
                self.logger.info(f"Aggressive DCA for {pair}: RSI={rsi:.2f}, ADX={adx:.2f}, new_stake={new_stake:.2f}")
            elif rsi > self.rsi_upper_threshold:
                new_stake = trade.stake_amount * 0.3 * self.risk_factor.value * (1 + volatility_factor) * self.leverage_factor.value
                self.logger.info(f"Conservative DCA for {pair}: RSI={rsi:.2f}, new_stake={new_stake:.2f}")
            else:
                new_stake = trade.stake_amount * 0.4 * self.risk_factor.value * (1 + volatility_factor) * self.leverage_factor.value
                self.logger.info(f"Standard DCA for {pair}: RSI={rsi:.2f}, new_stake={new_stake:.2f}")

            if total_position + new_stake > trade.stake_amount * 3 * self.risk_factor.value * self.leverage_factor.value:
                self.logger.info(f"Total position for {pair} exceeds {3 * self.risk_factor.value}x initial stake. Skipping DCA.")
                return None

            min_stake = max(min_stake, 1.0)
            if new_stake < min_stake:
                self.logger.warning(f"DCA stake {new_stake} for {pair} below min_stake {min_stake}. Adjusting.")
                new_stake = min_stake
            if new_stake > max_stake:
                self.logger.warning(f"DCA stake {new_stake} for {pair} exceeds max_stake {max_stake}. Adjusting.")
                new_stake = max_stake

            available_balance = self.wallets.get_free("USDT")
            if new_stake > available_balance:
                self.logger.warning(f"Cannot perform DCA for {pair}: new_stake {new_stake} exceeds available balance {available_balance}.")
                return None

            self.logger.info(f"DCA for {pair}: close={close:.6f}, avg_price={avg_price:.6f}, new_stake={new_stake:.2f}, price_drop={price_drop:.2%}, ADX={adx:.2f}, MACD={macd:.6f}")
            return new_stake

        return None

    def custom_roi(self, trade: Trade, current_profit: float, current_time: datetime, **kwargs) -> Dict[str, float]:
        time_open = (current_time - trade.open_date).total_seconds() / 3600  # Тривалість угоди в годинах
        roi_multiplier = 0.8 + 0.2 * self.leverage_factor.value / 5.0  # Адаптація ROI до кредитного плеча
        if time_open < 1:  # Менше 1 години
            dynamic_roi = max(0.002, 0.003 * self.risk_factor.value * (1 + current_profit) * roi_multiplier)
        elif time_open < 6:  # 1-6 годин
            dynamic_roi = max(0.004, 0.006 * self.risk_factor.value * (1 + current_profit * 1.5) * roi_multiplier)
        elif time_open < 24:  # 6-24 години
            dynamic_roi = max(0.006, 0.008 * self.risk_factor.value * (1 + current_profit * 2) * roi_multiplier)
        else:  # Більше 24 годин
            dynamic_roi = max(0.004, 0.005 * self.risk_factor.value * (1 + current_profit * 2.5) * roi_multiplier)
        
        self.logger.info(f"Trade {trade.pair} open for {time_open:.2f} hours, current profit {current_profit:.2%}, setting ROI to {dynamic_roi:.2%}")
        return {"0": dynamic_roi}

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe.iloc[-1]["atr"]
        # Динамічний стоп-лосс із урахуванням кредитного плеча
        dynamic_stoploss = -1.5 * atr / current_rate * self.risk_factor.value / self.leverage_factor.value
        return max(self.stoploss / self.leverage_factor.value, dynamic_stoploss)

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, **kwargs) -> float:
        return self.leverage_factor.value