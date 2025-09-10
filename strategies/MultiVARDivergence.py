#Steven's MultiVARDivergence Strategy

# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib







# This class is a sample. Feel free to customize it.
class MultiVARDivergence(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "30": 0.05,
        "45": 0.03,
        "60": 0.01,
        "75": 0.005,
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1

    # Trailing stoploss
    trailing_stop = True
    # trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03 # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = "15m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "entry": "market",# limit,market
        "exit": "market",# limit,market
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}


    #Not Implemented
    plot_config = {
    }

    ######################VAL##########################
    prd = 5  # Pivot Period
    source = "Close"  # Source for Pivot Points: "Close" or "High/Low"
    searchdiv = "Regular"  # Divergence Type: "Regular", "Hidden", "Regular/Hidden"
    maxpp = 10  # Maximum Pivot Points to Check
    maxbars = 100  # Maximum Bars to Check
    dontconfirm = False  # Don't Wait for Confirmation

    # Indicator toggles
    calcmacd = True  # MACD
    calcmacda = True  # MACD Histogram
    calcrsi = True  # RSI
    calcstoc = True  # Stochastic
    calccci = True  # CCI
    calcmom = True  # Momentum
    calcobv = True  # OBV
    calcvwmacd = True  # VWmacd
    calccmf = True  # Chaikin Money Flow
    calcmfi = True  # Money Flow Index
    calcext = False  # Check External Indicator

    # External indicator source
    externalindi = "close"  # Use string to represent the series name or function name
    ######################END VAL##########################

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """


        return []   
    def compute_pivots(self,df: pd.DataFrame, prd: int):
        """
        Identify pivot highs and lows in the DataFrame.
        Returns:
        ph_idx: list of indices of pivot highs
        ph_vals: list of high values at those pivots
        pl_idx: list of indices of pivot lows
        pl_vals: list of low values at those pivots
        """
        highs = df['high'].values
        lows  = df['low'].values
        n = len(df)
        ph_idx, pl_idx = [], []
        ph_vals, pl_vals = [], []
        for i in range(prd, n-prd):
            window_high = highs[i-prd:i+prd+1]
            window_low  = lows[i-prd:i+prd+1]
            if highs[i] == window_high.max():
                ph_idx.append(i)
                ph_vals.append(highs[i])
            if lows[i] == window_low.min():
                pl_idx.append(i)
                pl_vals.append(lows[i])
        return ph_idx, ph_vals, pl_idx, pl_vals


    def positive_regular_positive_hidden_divergence(
        self,
        df: pd.DataFrame,
        src: np.ndarray,
        cond_flag: bool,
        prd: int = 5,
        maxpp: int = 10,
        maxbars: int = 100,
        dontconfirm: bool = False,
        cond:int = 1
    ) -> np.ndarray:
        """
        For each bar, detect positive regular (cond=1) or positive hidden (cond=2) divergence.
        Returns an array of lookback lengths (0 if no divergence).
        """
        
        n = len(df)
        result = np.zeros(n, dtype=int)

        # compute pivot lows
        ph_idx, ph_vals, pl_idx, pl_vals = self.compute_pivots(df, prd)
        
        for i in range(n):
            if not dontconfirm:
                # i<1 时无法访问 i-1，直接跳过
                if i < 1:
                    continue
                # 两个条件都不成立就跳过
                if not (src[i] > src[i-1] or df['close'].iat[i] > df['close'].iat[i-1]):
                    continue
            if not cond_flag:
                continue
            # skip last candle if confirming
            start = 0 if dontconfirm else 1
            if not dontconfirm and i < 1:
                continue
            # current src value and price

            if cond == 1 and not cond_flag:
                continue
            # iterate past pivot lows
            count = 0

            # 先取出最后 maxpp 个，再通过 [::-1] 反转顺序
            recent_idx  = pl_idx[-maxpp:][::-1]
            recent_vals = pl_vals[-maxpp:][::-1]

            for j, (pi, pv) in enumerate(zip(recent_idx, recent_vals)):
                 # 计算当前 i 与枢轴点 pi 的回溯长度
                length = i - pi + prd

                 # —— 1. 如果枢轴点无效（pi==0）或长度超上限，就跳出整个循环 ——  
                if pi == 0 or length > maxbars:
                    break

                if length <= 5:
                    # Pine 是 break，但这里我们想跳过这个枢轴，继续下一个
                    continue  

                # 指标和价格的“起点”
                v_start = src[i - start]
                p_start = df['close'].iat[i - start]

                # 指标和价格的“枢轴点”  
                v_prev = src[i - length]
                p_prev = pl_vals[-maxpp:][::-1][j]  

                # —— 新增：计算两条斜率 ——  
                # 注意：斜率分母是 bars 数 = (length - start)
                slope1 = (v_start - v_prev) / (length - start)
                slope2 = (p_start - p_prev) / (length - start)

                # 初始化“虚拟趋势线”  
                virtual_line1 = v_start - slope1
                virtual_line2 = p_start - slope2

                # 在两端之间的每根 K 线都要检验不穿透
                arrived = True
                # 中间的索引，从（i-length+start+1）到（i-start-1）  
                for k in range(i - length + start + 1, i - start):
                    # 如果任一时刻指标 or 收盘价 跌破各自的“虚拟线”，就判为失败
                    if src[k] < virtual_line1 or df['close'].iat[k] < virtual_line2:
                        arrived = False
                        break
                    # 虚拟线往前推进一个 bar
                    virtual_line1 -= slope1
                    virtual_line2 -= slope2

                if arrived:
                    result[i] = length
                    break  # 找到一个就退出枢轴循环                
        return result


    def negative_regular_negative_hidden_divergence(
            self,
            df: pd.DataFrame,
            src: np.ndarray,
            cond_flag: bool,
            prd: int = 5,
            maxpp: int = 10,
            maxbars: int = 100,
            dontconfirm: bool = False,
            cond: int = 1
        ) -> np.ndarray:
        n = len(df)
        result = np.zeros(n, dtype=int)

        # 计算枢轴：ph_idx/ph_vals 是 pivot highs，高点索引和值
        ph_idx, ph_vals, pl_idx, pl_vals = self.compute_pivots(df, prd)

        for i in range(n):
            # —— Pine 中：if dontconfirm or src< src[1] or close< close[1] —— 
            if not dontconfirm:
                # i<1 时无法访问 i-1
                if i < 1:
                    continue
                # 既不是指标创新低，也不是收盘价创新低，就跳过
                if not (src[i] < src[i-1] or df['close'].iat[i] < df['close'].iat[i-1]):
                    continue

            # 用户自己设的开关
            if not cond_flag:
                continue

            # 确定 start（是否跳过当前 bar）
            start = 0 if dontconfirm else 1
            if not dontconfirm and i < 1:
                continue

            # 取最近 maxpp 个 pivot highs，反转为“最新→最旧”
            recent_idx  = ph_idx[-maxpp:][::-1]
            recent_vals = ph_vals[-maxpp:][::-1]

            for j, (pi, pv) in enumerate(zip(recent_idx, recent_vals)):
                # 回溯长度
                length = i - pi + prd

                # Pine: if pi==0 or length>maxbars then break 整个循环
                if pi == 0 or length > maxbars:
                    break

                # 太短就跳过本次枢轴
                if length <= 5:
                    continue

                # 起点指标 & 价格
                v_start = src[i - start]
                p_start = df['close'].iat[i - start]

                # 枢轴点指标 & 价格
                v_prev = src[i - length]
                p_prev = pv

                # 分两种背离类型判断
                # 负常规：指标新低(price低点)→价格不创新低  
                #    cond==1 时，要求 v_start < v_prev 且 p_start > p_prev
                if cond == 1 and not (v_start < v_prev and p_start > p_prev):
                    continue
                # 负隐藏：指标回升(价格创新低)→价格创新低  
                #    cond==2 时，要求 v_start > v_prev 且 p_start < p_prev
                if cond == 2 and not (v_start > v_prev and p_start < p_prev):
                    continue

                # —— 计算两条斜率 & 虚拟趋势线 —— 
                slope1 = (v_start - v_prev) / (length - start)
                slope2 = (p_start - p_prev) / (length - start)
                virtual1 = v_start - slope1
                virtual2 = p_start - slope2

                # 在两端之间的每根 K 线都要检验不穿透
                arrived = True
                # k 从 (i-length+start+1) 到 (i-start-1)
                for k in range(i - length + start + 1, i - start):
                    if src[k] > virtual1 or df['close'].iat[k] > virtual2:
                        arrived = False
                        break
                    virtual1 -= slope1
                    virtual2 -= slope2

                if arrived:
                    result[i] = length
                    break  # 找到一个就退出枢轴循环

        return result


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        df = dataframe.copy()

        # ——— 1. 计算所有指标 ——————————————————————————
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['rsi']      = ta.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = ta.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        df['cci']      = ta.CCI(df['high'], df['low'], df['close'], timeperiod=10)
        df['momentum']= ta.MOM(df['close'], timeperiod=10)
        df['obv']     = ta.OBV(df['close'], df['volume'])
        # VWMA fast/slow & VW-MACD
        df['vwma_fast'] = (df['close'] * df['volume']).rolling(12).sum() \
                         / df['volume'].rolling(12).sum()
        df['vwma_slow'] = (df['close'] * df['volume']).rolling(26).sum() \
                         / df['volume'].rolling(26).sum()
        df['vwmacd']    = df['vwma_fast'] - df['vwma_slow']
        # CMF(21)
        cmf_num = ((df['close'] - df['low']) - (df['high'] - df['close'])) \
                  / (df['high'] - df['low']) * df['volume']
        df['cmf'] = cmf_num.rolling(21).sum() / df['volume'].rolling(21).sum()
        # MFI(14)
        df['mfi']     = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

        # # ——— 2. 计算枢轴点 ————————————————————————————
        # # 计算枢轴点
        # ph_idx, ph_vals, pl_idx, pl_vals = self.compute_pivots(df, self.prd)

        # 一次性把所有指标的原始数组装进来
        indicators = {
            'MACD': df['macd'].values,
            'Hist': df['macd_hist'].values,
            'RSI':  df['rsi'].values,
            'Stoch_K': df['stoch_k'].values,
            'Stoch_D': df['stoch_d'].values,
            'CCI':  df['cci'].values,
            'Momentum': df['momentum'].values,
            'OBV': df['obv'].values,
            'VWMACD': df['vwmacd'].values,
            'CMF': df['cmf'].values,
            'MFI': df['mfi'].values,
        }
        df['Divergence_postive_tags']=""
        # 分别计算正向常规背离和正向隐藏背离
        pos_reg = {}
        pos_hid = {}
        for name, arr in indicators.items():
            pos_reg[name] = self.positive_regular_positive_hidden_divergence(
                df, arr,
                cond_flag=True,    # 启用背离检测
                prd=self.prd,
                maxpp=self.maxpp,
                maxbars=self.maxbars,
                dontconfirm=self.dontconfirm,
                cond=1             # **这里传 cond=1** 只检测正向常规背离
            )
            pos_hid[name] = self.positive_regular_positive_hidden_divergence(
                df, arr,
                cond_flag=True,
                prd=self.prd,
                maxpp=self.maxpp,
                maxbars=self.maxbars,
                dontconfirm=self.dontconfirm,
                cond=2             # **这里传 cond=2** 只检测正向隐藏背离
            )

        neg_reg = {}
        neg_hid = {}
        for name, arr in indicators.items():
            neg_reg[name] = self.negative_regular_negative_hidden_divergence(
                df, arr,
                cond_flag=True,    # 启用背离检测
                prd=self.prd,
                maxpp=self.maxpp,
                maxbars=self.maxbars,
                dontconfirm=self.dontconfirm,
                cond=1             # **这里传 cond=1** 只检测负向常规背离
            )
            neg_hid[name] = self.negative_regular_negative_hidden_divergence(
                df, arr,
                cond_flag=True,
                prd=self.prd,
                maxpp=self.maxpp,
                maxbars=self.maxbars,
                dontconfirm=self.dontconfirm,
                cond=2             # **这里传 cond=2** 只检测负向隐藏背离
            )


        for name in indicators.keys():
            df[f'{name}_pos_reg_len'] = pos_reg[name]
            df[f'{name}_pos_hid_len'] = pos_hid[name]
            df[f'{name}_neg_reg_len'] = neg_reg[name]
            df[f'{name}_neg_hid_len'] = neg_hid[name]

        # —— 2. 为每个指标生成“任意正背离”“任意负背离”标志 ——  
        for name in indicators.keys():
            df[f'{name}_pos_any'] = (
                (df[f'{name}_pos_reg_len'] > 0) |
                (df[f'{name}_pos_hid_len'] > 0)
            )
            df[f'{name}_neg_any'] = (
                (df[f'{name}_neg_reg_len'] > 0) |
                (df[f'{name}_neg_hid_len'] > 0)
            )

        # 合并到 DataFrame  
        df['pos_reg_any'] = np.any([v > 0 for v in pos_reg.values()], axis=0)
        df['pos_hid_any'] = np.any([v > 0 for v in pos_hid.values()], axis=0)
        df['neg_reg_any'] = np.any([v > 0 for v in neg_reg.values()], axis=0)
        df['neg_hid_any'] = np.any([v > 0 for v in neg_hid.values()], axis=0)

        return df
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        df = dataframe.copy()

        buy_signal = (
            (df['volume'] > 0) &
            (df['pos_reg_any'])    # | df['pos_hid_any'])
        )

        df.loc[buy_signal, 'enter_long'] = 1
        for i in ['MACD','Hist','RSI','Stoch_K','Stoch_D','CCI','Momentum','OBV','VWMACD','CMF','MFI']:
            if(np.any(df[f'{i}_pos_reg_len'])):
                df.loc[buy_signal,'enter_tag'] += i+'/'

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # df = dataframe.copy()

        # sell_signal = (
        #     (df['volume'] > 0) &
        #     (df['neg_reg_any'])    # | df['neg_hid_any'])
        # )

        # df.loc[sell_signal, 'exit_long'] = 1
        # for i in ['MACD','Hist','RSI','Stoch_K','Stoch_D','CCI','Momentum','OBV','VWMACD','CMF','MFI']:
        #     if(np.any(df[f'{i}_neg_reg_len'])):
        #         df.loc[sell_signal,'exit_tag'] += i+'/'

        # return df
    
        

        
