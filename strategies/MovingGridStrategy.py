from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import logging
from typing import Optional
import numpy as np
import talib.abstract as ta
import pandas as pd

class MovingGridStrategy(IStrategy):
    # ===== 核心参数 =====
    timeframe = '5m'
    use_custom_stoploss = True
    position_adjustment_enable = True
    
    # 风险控制
    max_open_trades = 1
    stoploss = -0.99
    minimal_roi = {"0": 10}
    
    # 网格参数
    grid_step_base = 0.5 / 100  # 基础网格间距 0.5%
    grid_orders = 5             # 每个方向网格数量
    stake_amount = 10           # 每格金额 10U
    
    # 网格移动参数
    move_threshold = 1 / 100    # 价格移动1%触发网格移动
    
    # 订单设置
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    process_only_new_candles = True
    use_exit_signal = False
    can_short = True  # 允许做空

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # 自定义日志格式 - 简化版本
        import logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 更简洁的日志格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 降低日志级别为INFO
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
        
        # 其他模块日志级别设置保持不变
        logging.getLogger('freqtrade').setLevel(logging.INFO)
        logging.getLogger('freqtrade.worker').setLevel(logging.INFO)
        logging.getLogger('freqtrade.strategy').setLevel(logging.INFO)  # 降低级别
        logging.getLogger('ccxt').setLevel(logging.INFO)  # 降低级别

    def informative_pairs(self):
        # 添加4小时时间框架用于趋势判断
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "4h") for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        
        # 添加波动率指标 (ATR)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # 计算ATR均值用于后续自适应网格
        dataframe['atr_ma'] = dataframe['atr'].rolling(window=50).mean()
        
        # 添加RSI指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # 获取4小时数据用于趋势判断
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe="4h")
        
        # 计算4小时EMA指标
        if not informative.empty:
            # 计算EMA指标
            informative['ema8'] = ta.EMA(informative, timeperiod=8)
            informative['ema34'] = ta.EMA(informative, timeperiod=34)
            
            # 简化合并逻辑 - 仅保留最新值
            last_informative = informative.iloc[-1]
            dataframe['ema8_4h'] = last_informative['ema8']
            dataframe['ema34_4h'] = last_informative['ema34']
            
            # 计算趋势方向
            dataframe['trend_direction'] = np.where(
                dataframe['ema8_4h'] > dataframe['ema34_4h'], 
                1,  # 上升趋势
                np.where(
                    dataframe['ema8_4h'] < dataframe['ema34_4h'], 
                    -1,  # 下降趋势
                    0  # 中性
                )
            )
        else:
            dataframe['trend_direction'] = 0
            self.logger.warning(f"无法获取 {pair} 的4小时数据")
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        
        # 初始化信号列
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # 检查是否有足够数据计算条件
        if len(dataframe) < 50:  # 增加到50确保有足够ATR数据
            if len(dataframe) < 20:
                self.logger.warning(f"数据不足 ({len(dataframe)}条)，跳过入场信号生成")
            return dataframe
        
        # 计算ATR是否上升
        atr_rising = (dataframe['atr'] > dataframe['atr'].shift(1))
        
        # 计算成交量是否高于20期均线
        volume_ma = dataframe['volume'].rolling(window=20).mean()
        volume_above_ma = (dataframe['volume'] > volume_ma)
        
        # 开仓条件：RSI超卖 + 波动率适中 + 符合趋势方向
        long_condition = (
            (dataframe['rsi'] < 30) &
            atr_rising &
            volume_above_ma &
            (dataframe['trend_direction'] == 1)  # 上升趋势才做多
        )
        
        short_condition = (
            (dataframe['rsi'] > 70) &  # RSI超买
            atr_rising &
            volume_above_ma &
            (dataframe['trend_direction'] == -1)  # 下降趋势才做空
        )
        
        dataframe.loc[long_condition, 'enter_long'] = 1
        dataframe.loc[short_condition, 'enter_short'] = 1
        
        # 记录入场条件详情
        last_row = dataframe.iloc[-1]
        trend_dir = last_row['trend_direction']
        trend_text = "上升" if trend_dir == 1 else "下降" if trend_dir == -1 else "中性"
        
        # 获取条件状态
        atr_change = atr_rising.iloc[-1]
        volume_condition_met = volume_above_ma.iloc[-1]
        
        # 修改后的日志输出
        if long_condition.iloc[-1] or short_condition.iloc[-1]:
            self.logger.info(f"入场 | {pair} | "
                            f"趋势: {trend_text} | "
                            f"ATR↑: {'是' if atr_change else '否'} | "
                            f"多: {'是' if long_condition.iloc[-1] else '否'} | "
                            f"空: {'是' if short_condition.iloc[-1] else '否'} | "
                            f"RSI↓: {'是' if last_row['rsi'] < 30 else '否'} | "
                            f"RSI↑: {'是' if last_row['rsi'] > 70 else '否'} | "
                            f"量↑: {'是' if volume_condition_met else '否'}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """自定义止损逻辑：价格偏离基准价10%时强制止损"""
        if hasattr(trade, 'grid_data'):
            base_price = trade.grid_data['base_price']
            price_diff = abs(current_rate - base_price) / base_price
            
            # 全局止损：偏离基准价10%
            if price_diff > 0.10:
                self.logger.warning(f"⚠️ 全局止损触发 | {pair} | "
                                   f"偏离: {price_diff*100:.2f}% | "
                                   f"基准价: {base_price:.4f} | 当前价: {current_rate:.4f}")
                return -1  # 全额止损
        
        return 1  # 默认禁用常规止损

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, 
                            time_in_force: str, current_time: datetime, side: str, **kwargs) -> bool:
        """开仓时初始化网格数据"""
        trade = kwargs.get('trade')
        if trade and not hasattr(trade, 'grid_data'):
            # 获取当前趋势方向
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                trend_direction = dataframe['trend_direction'].iloc[-1]
                # 获取当前ATR数据用于自适应网格
                current_atr = dataframe['atr'].iloc[-1]
                atr_ma = dataframe['atr_ma'].iloc[-1]
            else:
                trend_direction = 1 if side == "long" else -1
                current_atr = None
                atr_ma = None
            
            trade.grid_data = {
                "grid_levels": [],
                "base_price": rate,
                "highest_price": rate,
                "lowest_price": rate,
                "total_buys": 0,
                "total_sells": 0,
                "last_move_time": current_time,
                "last_trade_time": current_time - timedelta(minutes=10),  # 初始化上次交易时间
                "trend_direction": trend_direction,  # 记录开仓时的趋势方向
                "current_atr": current_atr,
                "atr_ma": atr_ma
            }
            
            self.logger.info(f"📊 [网格初始化] {pair} | {'多单' if side == 'long' else '空单'} | "
                            f"基准价: {rate:.4f} | 数量: {amount:.6f} | "
                            f"趋势方向: {'上升' if trend_direction == 1 else '下降'}")
        return True

    def adjust_trade_position(self, trade, current_time: datetime,
                               current_rate: float, current_profit: float, **kwargs):
        """移动网格核心逻辑 - 增加详细诊断日志"""
        pair = trade.pair
        
        # 检查上次交易时间，防止过于频繁交易
        time_since_last_trade = (current_time - trade.grid_data["last_trade_time"]).total_seconds()
        if time_since_last_trade < 60:  # 至少60秒间隔
            self.logger.debug(f"交易过于频繁 | {pair} | 上次交易: {time_since_last_trade:.1f}秒前")
            return None
        
        if not hasattr(trade, 'grid_data'):
            self.logger.warning(f"⚠️ {pair} 没有网格数据，跳过调整")
            return None
            
        grid_data = trade.grid_data
        grid_levels = grid_data["grid_levels"]
        base_price = grid_data["base_price"]
        highest_price = grid_data["highest_price"]
        lowest_price = grid_data["lowest_price"]
        trend_direction = grid_data.get("trend_direction", 1)  # 默认为上升趋势
        
        # 记录当前网格状态
        self.logger.debug(f"网格状态 | 基准价: {base_price:.4f} | 最高价: {highest_price:.4f} | "
                         f"最低价: {lowest_price:.4f} | 已触发网格: {len(grid_levels)}")
        
        # 更新价格范围
        price_updated = False
        if current_rate > highest_price:
            self.logger.debug(f"更新最高价 | 旧: {highest_price:.4f} → 新: {current_rate:.4f}")
            highest_price = current_rate
            grid_data["highest_price"] = highest_price
            price_updated = True
        if current_rate < lowest_price:
            self.logger.debug(f"更新最低价 | 旧: {lowest_price:.4f} → 新: {current_rate:.4f}")
            lowest_price = current_rate
            grid_data["lowest_price"] = lowest_price
            price_updated = True
        
        # 计算价格变化百分比
        price_diff = abs(current_rate - base_price) / base_price
        self.logger.debug(f"价格变化检查 | 当前价: {current_rate:.4f} | 基准价: {base_price:.4f} | "
                         f"变化率: {price_diff*100:.2f}% | 阈值: {self.move_threshold*100:.2f}%")
        
        moved = False
        # 移动条件检查：价格变化超过阈值
        if price_diff >= self.move_threshold:
            # 防止过于频繁移动：至少间隔1根K线
            time_since_last_move = (current_time - grid_data["last_move_time"]).total_seconds() / 60
            self.logger.debug(f"移动条件满足 | 时间差: {time_since_last_move:.1f}分钟 | 阈值: >5分钟")
            
            if time_since_last_move > 5:  # 5分钟
                moved = True
                base_price = current_rate
                grid_levels = []
                grid_data["base_price"] = base_price
                grid_data["grid_levels"] = grid_levels
                grid_data["last_move_time"] = current_time
                
                self.logger.info(f"🔄 [网格移动] {pair} | 变化: {price_diff*100:.2f}% | 新基准价: {base_price:.4f}")
        
        # 防止过度交易：网格已饱和时暂停交易
        if len(grid_levels) >= self.grid_orders * 2:
            self.logger.warning(f"⚠️ 网格已饱和 | {pair} | 已触发网格数: {len(grid_levels)} | 最大网格数: {self.grid_orders*2}")
            return None
        
        # 计算动态网格间距
        current_atr = grid_data.get("current_atr")
        atr_ma = grid_data.get("atr_ma")
        
        if current_atr is not None and atr_ma is not None and atr_ma > 0:
            atr_ratio = current_atr / atr_ma
            # 限制调整范围在0.8到1.2之间
            atr_adjustment = max(0.8, min(atr_ratio, 1.2))
            dynamic_grid_step = self.grid_step_base * atr_adjustment
        else:
            dynamic_grid_step = self.grid_step_base
        
        self.logger.debug(f"网格间距 | 基础: {self.grid_step_base*100:.2f}% | "
                         f"动态: {dynamic_grid_step*100:.2f}% | "
                         f"ATR比率: {atr_ratio if current_atr else 'N/A':.2f}")
        
        # 生成网格价格 - 根据趋势方向调整
        new_levels = []
        for i in range(1, self.grid_orders + 1):
            if trend_direction == 1:  # 上升趋势 - 做多网格
                # 下方买入网格
                buy_price = round(base_price * (1 - i * dynamic_grid_step), 6)
                new_levels.append(buy_price)
                
                # 上方卖出网格
                sell_price = round(base_price * (1 + i * dynamic_grid_step), 6)
                new_levels.append(sell_price)
            else:  # 下降趋势 - 做空网格
                # 上方卖出网格 (做空)
                sell_price = round(base_price * (1 + i * dynamic_grid_step), 6)
                new_levels.append(sell_price)
                
                # 下方买入网格 (平空)
                buy_price = round(base_price * (1 - i * dynamic_grid_step), 6)
                new_levels.append(buy_price)
        
        # 记录网格状态
        buy_levels = [p for p in new_levels if p < current_rate]
        sell_levels = [p for p in new_levels if p > current_rate]
        
        # 减少日志频率
        if moved or price_updated:
            self.logger.info(f"📈 [网格状态] {pair} | 方向: {'多' if trend_direction == 1 else '空'} | "
                            f"基准价: {base_price:.4f} | 当前价: {current_rate:.4f}")
            self.logger.debug(f"🔽 买入网格 ({len(buy_levels)}个): {[f'{p:.4f}' for p in buy_levels]}")
            self.logger.debug(f"🔼 卖出网格 ({len(sell_levels)}个): {[f'{p:.4f}' for p in sell_levels]}")
            self.logger.debug(f"已触发网格点: {[f'{p:.4f}' for p in grid_levels]}")
        
        # 买入网格逻辑
        for buy_price in buy_levels:
            if buy_price not in grid_levels:
                self.logger.debug(f"检查买入网格 | 价格: {buy_price:.4f} | 当前价: {current_rate:.4f} | "
                                 f"条件: {current_rate >= buy_price}")
                
                if current_rate >= buy_price:
                    grid_levels.append(buy_price)
                    grid_data["total_buys"] += 1
                    grid_data["grid_levels"] = grid_levels
                    grid_data["last_trade_time"] = current_time  # 更新最后交易时间
                    trade.grid_data = grid_data
                    
                    action = "加仓" if trend_direction == 1 else "平空"
                    self.logger.info(f"💰 [{action}] {pair} | 价格: {buy_price:.4f} | 金额: {self.stake_amount} USDT")
                    return -self.stake_amount
        
        # 卖出网格逻辑
        for sell_price in sell_levels:
            if sell_price not in grid_levels:
                self.logger.debug(f"检查卖出网格 | 价格: {sell_price:.4f} | 当前价: {current_rate:.4f} | "
                                 f"条件: {current_rate <= sell_price}")
                
                if current_rate <= sell_price:
                    grid_levels.append(sell_price)
                    grid_data["total_sells"] += 1
                    grid_data["grid_levels"] = grid_levels
                    grid_data["last_trade_time"] = current_time  # 更新最后交易时间
                    trade.grid_data = grid_data
                    
                    action = "减仓" if trend_direction == 1 else "加空"
                    self.logger.info(f"💸 [{action}] {pair} | 价格: {sell_price:.4f} | 金额: {self.stake_amount} USDT")
                    return self.stake_amount
        
        # 保存状态（无交易时）
        trade.grid_data = grid_data
        
        # 记录未触发原因
        if not buy_levels and not sell_levels:
            self.logger.warning(f"⚠️ 没有生成任何网格点 | 基准价: {base_price:.4f} | 当前价: {current_rate:.4f}")
        elif not moved and not price_updated and not grid_levels:
            self.logger.debug("✅ 网格状态正常，但当前价格未触及任何新网格点")
            
        return None

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                          entry_tag: Optional[str], side: str, **kwargs) -> float:
        """获取买入限价单价格"""
        try:
            order_book = self.dp.orderbook(pair, 1)
            if order_book:
                if side == "long" and 'asks' in order_book and len(order_book['asks']) > 0:
                    ask_price = order_book["asks"][0][0]
                    self.logger.debug(f"订单簿买入价 | 卖一价: {ask_price:.4f} | 提议价: {proposed_rate:.4f}")
                    return ask_price
                elif side == "short" and 'asks' in order_book and len(order_book['asks']) > 0:
                    ask_price = order_book["asks"][0][0]
                    self.logger.debug(f"做空订单价 | 卖一价: {ask_price:.4f} | 提议价: {proposed_rate:.4f}")
                    return ask_price
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}", exc_info=True)
        return proposed_rate

    def custom_exit_price(self, pair: str, trade, current_time: datetime, proposed_rate: float,
                          exit_tag: Optional[str], side: str, **kwargs) -> float:
        """获取卖出限价单价格"""
        try:
            order_book = self.dp.orderbook(pair, 1)
            if order_book:
                if side == "long" and 'bids' in order_book and len(order_book['bids']) > 0:
                    bid_price = order_book["bids"][0][0]
                    self.logger.debug(f"订单簿卖出价 | 买一价: {bid_price:.4f} | 提议价: {proposed_rate:.4f}")
                    return bid_price
                elif side == "short" and 'bids' in order_book and len(order_book['bids']) > 0:
                    bid_price = order_book["bids"][0][0]
                    self.logger.debug(f"平空订单价 | 买一价: {bid_price:.4f} | 提议价: {proposed_rate:.4f}")
                    return bid_price
        except Exception as e:
            self.logger.error(f"获取订单簿失败: {e}", exc_info=True)
        return proposed_rate

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           **kwargs) -> float:
        """资金管理：保留安全缓冲"""
        free_usdt = self.wallets.get_free('USDT')
        if free_usdt < 10:
            self.logger.warning(f"⚠️ 资金不足! 可用USDT: {free_usdt:.2f}, 保留10 USDT安全缓冲")
            return 0
        return self.stake_amount

    def bot_start(self, **kwargs) -> None:
        """机器人启动时记录配置"""
        self.logger.info("=== 网格策略启动 ===")
        self.logger.info(f"网格参数 | 基础间距: {self.grid_step_base*100:.2f}% | 数量: {self.grid_orders} | 每格金额: {self.stake_amount} USDT")
        self.logger.info(f"移动阈值: {self.move_threshold*100:.2f}% | 时间框架: {self.timeframe}")
        self.logger.info(f"最大开仓数: {self.max_open_trades} | 止损: 偏离基准价10%")
        self.logger.info(f"趋势判断时间框架: 4小时 | EMA8 vs EMA34 | 动态ATR调整")

    def bot_loop_start(self, **kwargs) -> None:
        """每次循环开始时记录状态"""
        # 减少日志频率，每小时记录一次
        current_time = datetime.now()
        if current_time.minute == 0 and current_time.second < 10:
            free_usdt = self.wallets.get_free('USDT')
            total_usdt = self.wallets.get_total('USDT')
            # self.logger.info(f"钱包状态 | 可用USDT: {free_usdt:.2f} | 总USDT: {total_usdt:.2f}")
