from freqtrade.strategy import IStrategy, IntParameter,RealParameter
from pandas import DataFrame
import talib.abstract as ta
from technical import qtpylib
import json
import os


class ImprovedSampleStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count: int = 50
   

   

    # 设置默认参数，会在 __init__ 中被 JSON 覆盖
    minimal_roi = {
        "0": 0.04,
        "30": 0.02,
        "60": 0.01
    }
    stoploss = -0.3

    buy_rsi = IntParameter(20, 50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(70, 90, default=80, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(60, 90, default=75, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    protection = RealParameter(0.001, 0.05, default=0.02, space="protection", optimize=True, load=True)
    
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)

        # 获取 json 路径
        json_path = os.path.join(os.path.dirname(__file__), 'ImprovedSampleStrategy.json')

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    strategy_config = json.load(f)
                    params = strategy_config.get("params", {})

                    # 设置 ROI 和 Stoploss
                    self.minimal_roi = params.get("roi", self.minimal_roi)
                    self.stoploss = params.get("stoploss", {}).get("stoploss", self.stoploss)

                    # 设置 RSI 参数默认值
                    self.buy_rsi.default = params.get("buy", {}).get("buy_rsi", self.buy_rsi.default)
                    self.exit_short_rsi.default = params.get("buy", {}).get("exit_short_rsi", self.exit_short_rsi.default)
                    self.sell_rsi.default = params.get("sell", {}).get("sell_rsi", self.sell_rsi.default)
                    self.short_rsi.default = params.get("sell", {}).get("short_rsi", self.short_rsi.default)
                        # 设置 protection 默认值（如果存在）
                    if "protection" in params:
                        self.protection.default = params["protection"].get("protection", self.protection.default)


                except Exception as e:
                    print(f" JSON 读取错误: {e}")
        else:
            print("未找到 JSON 配置文件，使用默认参数。")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)
        dataframe["cci"] = ta.CCI(dataframe)
        boll = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_upper"] = boll["upper"]
        dataframe["bb_middle"] = boll["mid"]
        dataframe["bb_lower"] = boll["lower"]
        dataframe["adx"] = ta.ADX(dataframe)
        macd = ta.MACD(dataframe)
        dataframe["macd_hist"] = macd["macdhist"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)
                & (dataframe["tema"] > dataframe["bb_middle"])
                & (dataframe["adx"] > 25)  # 调整ADX阈值
                & (dataframe["macd_hist"] > 0)
                & (dataframe["volume"] > dataframe["volume"].rolling(window=14).mean())  # 增加成交量条件
                & ((dataframe["rsi"] < self.buy_rsi.value + 5))

            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)
                & (dataframe["adx"] > 25)  # 调整ADX阈值
                & (dataframe["volume"] > dataframe["volume"].rolling(window=14).mean())  # 增加成交量条件
                & ((dataframe["rsi"] < self.buy_rsi.value + 5))
            ),
            "exit_long",
        ] = 1
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["bb_lower"]) &
                (dataframe["cci"] < -100) &
                (dataframe["adx"] > 25) &
                (dataframe["volume"] > dataframe["volume"].rolling(5).mean())
            ),
            "enter_long",
        ] = 1
        return dataframe
    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["future_profit"] = dataframe["close"].shift(-6) / dataframe["close"] - 1
        dataframe["future_up_2pct"] = (dataframe["future_profit"] > 0.02).astype(int)
        return dataframe
