from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade, Order
from pandas import DataFrame, Timestamp
import pandas as pd
import talib.abstract as ta
from datetime import datetime, timedelta
import logging

RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"
logger = logging.getLogger(__name__)


class DcaTpShort_3(IStrategy):
    timeframe = '30m'
    stoploss = -7
    can_short = True
    can_long = False
    use_exit_signal = False
    trailing_stop = False
    position_adjustment_enable = True

    minimal_roi = {"0": 777.0}
    minimal_roi_user_defined = {
        "300": 0.010, "290": 0.020, "280": 0.030, "270": 0.040, "260": 0.050,
        "250": 0.060, "240": 0.080, "230": 0.090, "220": 0.100, "210": 0.110,
        "200": 0.120, "190": 0.130, "180": 0.140, "160": 0.145, "140": 0.150,
        "120": 0.155, "100": 0.160, "80": 0.165, "60": 0.170, "50": 0.175,
        "40": 0.180, "30": 0.185, "20": 0.190, "10": 0.195, "0": 0.200,
    }

    def leverage(self, pair: str, **kwargs) -> float:
        return 20

    def on_trade_open(self, trade: Trade, **kwargs) -> None:
        flags = {
            'dca_count': 0, 'tp_count': 0, 'dca_done': False,
            'last_dca_candle': None, 'last_dca_time': None,
            'dca_reduce_done': False, 'open_reduce_done': False,
            'need_rebuy': False, 'last_tp_time': None,
            'low_margin_start': None, 'trend_level': 0,
            'top_added': False, 'bottom_reduced': False,
            'bb_added': False, 'pullback_ready_short': True,
            'trend_reset': False, 'last_trend_side': 'none',
            'last_fallback_price_short': None, 'fallback_repull_done_short': False
        }
        for k, v in flags.items():
            trade.set_custom_data(k, v)
        trade.set_custom_data('dynamic_avg_entry', trade.open_rate)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        upper, mid, lower = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_upperband'] = upper
        dataframe['bb_midband'] = mid
        dataframe['bb_lowerband'] = lower
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        dataframe['vol_ma20'] = dataframe['volume'].rolling(20).mean()
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], timeperiod=14)
        dataframe['atr_ma'] = dataframe['atr'].rolling(14).mean()

        df30 = self.dp.get_pair_dataframe(metadata['pair'], '30m')
        if not df30.empty:

            macd, macdsig, macdhist = ta.MACD(
                df30['close'], fastperiod=8, slowperiod=21, signalperiod=5
            )

            k, d = ta.STOCH(
                df30['high'], df30['low'], df30['close'],
                fastk_period=5, slowk_period=3, slowd_period=3
            )
            j = 3 * k - 2 * d

            ema9 = ta.EMA(df30['close'], timeperiod=9)
            ema21 = ta.EMA(df30['close'], timeperiod=21)
            ema99 = ta.EMA(df30['close'], timeperiod=99)
            adx = ta.ADX(df30['high'], df30['low'], df30['close'])

            series_map = {
                'macd_30': macd, 'macdsig_30': macdsig,
                'k_30': k, 'd_30': d, 'j_30': j,
                'ema9_30': ema9, 'ema21_30': ema21, 'ema99_30': ema99,
                'adx_30': adx
            }
            for name, series in series_map.items():
                dataframe[name] = pd.Series(series, index=df30.index).reindex(dataframe.index).ffill()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        short_cond1 = (
                (dataframe['macd_30'] < dataframe['macdsig_30']) &
                (dataframe['k_30'] < dataframe['d_30']) &
                (dataframe['adx_30'] > 25) &
                (dataframe['ema9_30'] < dataframe['ema21_30']) &
                (dataframe['ema21_30'] < dataframe['ema99_30'])
        )



        short_cond2 = (
                (dataframe['close'] > dataframe['bb_upperband']) &
                (dataframe['rsi'] > 65)
        )
        dataframe['enter_short'] = (short_cond1 | short_cond2).astype(int)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_short'] = 0
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, **kwargs) -> tuple[float, str] | None:
        if current_time.tzinfo:
            current_time = current_time.replace(tzinfo=None)
        open_time = trade.open_date_utc
        if open_time.tzinfo:
            open_time = open_time.replace(tzinfo=None)

        if trade.has_open_orders:
            return None
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if df.empty:
            return None
        candle_ts = pd.Timestamp(df.index[-1]).tz_localize(None).floor('min')
        last = df.iloc[-1]
        margin = float(trade.stake_amount)

        if not self.wallets:
            return None
        collateral = self.wallets.get_total('USDT')

        def collateral_add(frac: float) -> float:
            return collateral * frac

        price = last['close']
        lower = last['bb_lowerband']
        upper = last['bb_upperband']
        mid = last['bb_midband']



















































        low14 = df['close'].rolling(14).min().iat[-1]
        if (last['ema9_30'] < last['ema21_30'] < last['ema99_30']
                and last['close'] == low14):
            trade.set_custom_data('ref_low', float(low14))
            trade.set_custom_data('pullback_done_short', False)
        ref_low = trade.get_custom_data('ref_low')
        done_s = bool(trade.get_custom_data('pullback_done_short'))
        ready_s = bool(trade.get_custom_data('pullback_ready_short'))

        if (ref_low is not None and ready_s and not done_s
                and current_rate >= ref_low * 1.01
                and last['ema9_30'] < last['ema21_30']):
            amt = collateral_add(0.02)  # 反弹加仓参数
            trade.set_custom_data('pullback_done_short', True)
            trade.set_custom_data('pullback_ready_short', False)
            logger.info(
                f"{RED}[{trade.pair}] 趋势反弹加仓总资金 2%"
                f"低点={ref_low:.4f}, 当前价={current_rate:.4f}, "
                f"保证金={margin:.4f}, 加空={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'short_pullback_dca20'

        last_rsi2 = last['rsi']

        def get_cd(key, default=None):
            v = trade.get_custom_data(key)
            return default if v is None or (isinstance(v, str) and v.lower() == 'null') else v

        last_dca = get_cd('last_dca_candle')
        last_dca_ts = Timestamp(last_dca, unit='s') if isinstance(last_dca, (int, float)) else None
        if last_dca_ts != candle_ts:
            trade.set_custom_data('dca_done', False)
        dca_done = bool(get_cd('dca_done', False))
        avg = float(get_cd('dynamic_avg_entry', trade.open_rate))
        u = int(get_cd('dca_count', 0))
        threshold = avg * (1 + 0.01 + 0.01 * u)  # Dca加仓价格参数
        rsi_thresh = max(0, 65)  # RSI参数

        if not dca_done and current_rate >= threshold and last_rsi2 > rsi_thresh:
            amt = collateral_add(0.02)  # DCA加仓参数
            leverage = self.leverage(trade.pair)
            prev_qty = abs(float(trade.amount))
            prev_cost = prev_qty * avg
            added_qty = (abs(amt) * leverage) / current_rate
            new_avg = (prev_cost + abs(amt) * leverage) / (prev_qty + added_qty)
            trade.set_custom_data('dca_count', u + 1)
            trade.set_custom_data('dca_done', True)
            trade.set_custom_data('last_dca_candle', int(candle_ts.timestamp()))
            trade.set_custom_data('last_dca_time', int(current_time.timestamp()))
            trade.set_custom_data('dca_reduce_done', False)
            trade.set_custom_data('open_reduce_done', False)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dynamic_avg_entry', new_avg)
            logger.info(
                f"[{trade.pair}][浮亏 DCA 加仓] u={u}->{u + 1}, "
                f"{YELLOW}保证金={trade.stake_amount:.8f}{RESET}{RED}加仓={abs(amt):.8f}{RESET}, "
                f"{BLUE}新均价={new_avg:.8f}{RESET}"
            ),
            return amt, f"dca_u_short={u + 1}"

        n = int(trade.get_custom_data('tp_count') or 0)
        need_rebuy = bool(trade.get_custom_data('need_rebuy'))
        if need_rebuy:
            amt = collateral_add(0.05)  # 首次浮盈加仓参数
            tag = 'rebuy50_short'
            logger.info(
                f"[{trade.pair}][分批止盈】 加仓总资金 5% u={u}, n={n}, "
                f"{YELLOW}保证金={margin:.4f}{RESET}, {GREEN}加仓={abs(amt):.4f}{RESET}"
            )
            trade.set_custom_data('need_rebuy', False)
            trade.set_custom_data('dca_done', False)
            return amt, tag

        if n > 0 and current_profit < 0.01:
            pct = -min(1.0, 0.50 + 0.05 * n)
            amt = pct * margin
            trade.set_custom_data('last_fallback_price_short', price)
            trade.set_custom_data('fallback_ready_short', True)
            trade.set_custom_data('fallback_repull_done_short', False)
            logger.info(
                f"{YELLOW}[{trade.pair}] 止盈后回撤减仓{int(abs(pct) * 100)}%: "
                f"回撤价={price:.4f}, 减仓={abs(amt):.4f} USDT{RESET}"
            )
            trade.set_custom_data('dca_count', 0)
            trade.set_custom_data('tp_count', 0)
            trade.set_custom_data('dca_done', False)
            trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
            return amt, f"tp_fallback1%_{int(abs(pct) * 100)}%_short"

        last_fb = trade.get_custom_data('last_fallback_price_short')
        ready = bool(trade.get_custom_data('fallback_ready_short'))
        done = bool(trade.get_custom_data('fallback_repull_done_short'))

        if last_fb and ready and not done and price >= last_fb * 1.01:  # 反弹价格参数
            amt = collateral_add(0.02)  # 反弹加仓参数
            trade.set_custom_data('fallback_repull_done_short', True)
            trade.set_custom_data('fallback_ready_short', False)
            logger.info(
                f"{GREEN}[{trade.pair}] 回撤价反弹加空总资金 2%: "
                f"回撤价={last_fb:.4f}, 当前价={price:.4f}, "
                f"保证金={margin:.4f}, 加空={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'tp_repull20_short'

        last_tp = trade.get_custom_data('last_tp_time')
        base = datetime.fromtimestamp(last_tp) if last_tp else open_time
        elapsed = (current_time - base).total_seconds() / 60
        roi = 0.0
        for k, v in sorted(self.minimal_roi_user_defined.items(), key=lambda x: int(x[0]), reverse=True):
            if elapsed >= int(k):
                roi = v
                break
        if current_profit >= roi:
            if u > 0:
                pct = min(1.0, 0.50 + 0.05 * u)  # 浮亏止盈卖出参数
                amt = -pct * margin
                logger.info(
                    f"[{trade.pair}][浮亏 DCA 后止盈] u={u}, n={n}, {YELLOW}保证金={margin:.2f}{RESET},"
                    f"{GREEN}减仓={abs(amt):.2f}{RESET}"
                )
                trade.set_custom_data('dca_count', 0)
                trade.set_custom_data('tp_count', 0)
                trade.set_custom_data('dca_done', False)
                trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                return amt, f"tp_afterDCA_short_u{u}"
            else:
                if not last_tp or Timestamp(last_tp, unit='s').floor('T') != candle_ts:
                    amt = -0.30 * margin  # 浮盈止盈卖出参数
                    logger.info(
                        f"[{trade.pair}][浮盈减仓 卖30%→后续加仓 5%] u=0, n={n}->{n + 1}, "
                        f"{YELLOW}保证金={margin:.2f}{RESET}, {GREEN}减仓={abs(amt):.2f}{RESET}"
                    )
                    trade.set_custom_data('tp_count', n + 1)
                    trade.set_custom_data('dca_count', 0)
                    trade.set_custom_data('dca_done', False)
                    trade.set_custom_data('last_tp_time', int(current_time.timestamp()))
                    return amt, 'tp30'

        if trade.get_custom_data('top_added') and price < lower:
            trade.set_custom_data('top_added', False)
        if trade.get_custom_data('bottom_reduced') and price > upper:
            trade.set_custom_data('bottom_reduced', False)

        if not trade.get_custom_data('top_added') and last['j_30'] > 100 and last['rsi'] > 65:  # KDJ_J&Rsi参数
            trade.set_custom_data('top_added', True)
            amt = collateral_add(0.02)
            logger.info(
                f"{BLUE}[{trade.pair}] 抄顶加空总资金 2%: J={last['j_30']:.2f}, RSI={last['rsi']:.1f}, "
                f"保证金={margin:.4f}, 加空={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'top_add20_short'

        if not trade.get_custom_data('bottom_reduced') and current_profit > 0 and last['j_30'] < 0 and last[
            'rsi'] < 35:  # KDJ_J&Rsi参数
            trade.set_custom_data('bottom_reduced', True)
            amt = -0.5 * margin
            logger.info(
                f"{RED}[{trade.pair}] 逃底减仓50%: J={last['j_30']:.2f}, RSI={last['rsi']:.1f}, "
                f"保证金={margin:.4f}, 减仓={abs(amt):.4f} USDT{RESET}"
            )
            return amt, 'bottom_cover50_short'

        if trade.get_custom_data('bottom_reduced') and price >= mid:
            trade.set_custom_data('bottom_reduced', False)
            amt = collateral_add(0.02)  # 回落加仓参数
            logger.info(
                f"{GREEN}[{trade.pair}] 逃底反弹加仓总资金 2%: 当前价={price:.4f}, 布林中轨={mid:.4f}, "
                f"保证金={margin:.4f}, 加仓={amt:.4f} USDT{RESET}"
            )
            return amt, 'rebound_add20'





































        return None

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        tag = getattr(order, 'ft_order_tag', None)
        if tag == "tp30":
            trade.set_custom_data('need_rebuy', True)
            trade.set_custom_data('pullback_ready_short', True)
            trade.set_custom_data('last_tp_price', order.price)
            trade.set_custom_data('fallback_repull_done_short', False)

    def custom_stoploss(self, *args, **kwargs) -> float | None:
        return None

