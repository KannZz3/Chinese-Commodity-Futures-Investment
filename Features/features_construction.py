# Features/features_construction.py

import numpy as np
import pandas as pd

__all__ = ["build_price_features"]


# ============================
#  纯价格行为因子构建
# Price-action-only features
# ============================
def build_price_features(
    df,
    mom_windows=(2, 4, 8, 16),
    vol_windows=(2, 4, 8, 16, 32),
    ma_windows=(2, 4, 8, 16, 32),
    vol_ma_windows=(3, 5, 10, 20),
    hold_ma_windows=(3, 5, 10, 20),
):

    data = df.copy()

    # =========================
    # 1. Single-bar return & candlestick shape
    # =========================
    # 单根K线收益率：计算当前收盘价与前一天收盘价的百分比变化
    # Single-bar return: Percentage change between today's and previous day's close
    data["ret_1"] = data["close"].pct_change()

    # K线形态：计算实体、范围、上影线和下影线
    # Candlestick shape: Calculate body, range, upper and lower shadows
    data["body"] = data["close"] - data["open"]
    data["range"] = data["high"] - data["low"]
    data["upper_shadow"] = data["high"] - data[["open", "close"]].max(axis=1)
    data["lower_shadow"] = data[["open", "close"]].min(axis=1) - data["low"]

    # 各部分占比（归一化）
    # Ratios (normalized by range)
    data["body_pct"] = data["body"] / data["range"].replace(0, np.nan)
    data["upper_shadow_pct"] = data["upper_shadow"] / data["range"].replace(0, np.nan)
    data["lower_shadow_pct"] = data["lower_shadow"] / data["range"].replace(0, np.nan)

    # 多头/空头判定：收盘价高于开盘价为多头，反之为空头
    # Bullish/Bearish: Close > Open means bullish, else bearish
    data["is_bull"] = (data["close"] > data["open"]).astype(int)
    data["is_bear"] = (data["close"] < data["open"]).astype(int)

    # =========================
    # 2. Momentum, volatility & moving-average factors
    # =========================

    # -------- 1. Momentum --------
    for n in mom_windows:
        col = f"mom_{n}"
        # 动量：n日收盘价的百分比变化
        # Momentum: Percentage change in close over n days
        data[col] = data["close"].pct_change(n)

        # 动量变化速度（斜率）
        # Momentum slope: Percentage change of momentum
        data[f"{col}_slope"] = data[col].pct_change()

        # 多周期动量变化
        # Multi-period momentum slopes
        data[f"{col}_slope_3"] = data[col].pct_change(3)
        data[f"{col}_slope_5"] = data[col].pct_change(5)
        data[f"{col}_slope_10"] = data[col].pct_change(10)
        data[f"{col}_slope_25"] = data[col].pct_change(25)

        # 动量加速度（第二导数）
        # Momentum acceleration (second derivative)
        data[f"{col}_acc"] = data[col].diff().diff()

        # 动量角度：动量变化的角度，反映动量强度
        # Momentum angle slope: Angle of momentum change, reflecting momentum strength
        data[f"mom_{n}_angle"] = np.arctan(
            data[f"mom_{n}"].diff() / (data[f"mom_{n}"].shift(1).abs() + 1e-9)
        )

    # -------- 2. Volatility --------
    for n in vol_windows:
        col = f"vol_{n}"
        # 波动率：收益率的滚动标准差
        # Volatility: Rolling standard deviation of returns
        data[col] = data["ret_1"].rolling(n).std()

        # 波动率变化（斜率）
        # Volatility slope: Percentage change of volatility
        data[f"{col}_slope"] = data[col].pct_change()

        # 多周期波动率变化
        # Multi-period volatility slopes
        data[f"{col}_slope_3"] = data[col].pct_change(3)
        data[f"{col}_slope_5"] = data[col].pct_change(5)
        data[f"{col}_slope_10"] = data[col].pct_change(10)
        data[f"{col}_slope_25"] = data[col].pct_change(25)

        # 波动率加速度（第二导数）
        # Volatility acceleration (second derivative)
        data[f"{col}_acc"] = data[col].diff().diff()


    # -------- 3. Moving averages (MA) --------
    
    # WMA：加权移动平均线，赋予最近数据更大权重
    # WMA: Weighted Moving Average, giving more weight to recent data
    data["wma_5"] = data["close"].rolling(5).apply(lambda x: np.dot(x, np.arange(1, 6)) / 15, raw=False)
    data["wma_10"] = data["close"].rolling(10).apply(lambda x: np.dot(x, np.arange(1, 11)) / np.sum(np.arange(1, 11)), raw=False)
    for n in ma_windows:
        ma_col = f"ma_{n}"
        # 简单移动平均：n日收盘价的均值
        # Simple Moving Average (SMA): Mean of close price over n days
        data[ma_col] = data["close"].rolling(n).mean()
        data[f"close_ma_{n}"] = data["close"] / data[ma_col] - 1
        
        # 均线的变化速度（斜率）
        # Moving average slope: Percentage change in moving average
        data[f"{ma_col}_slope"] = data[ma_col].pct_change()

        # 多周期均线变化
        # Multi-period moving average slopes
        data[f"{ma_col}_slope_3"] = data[ma_col].pct_change(3)
        data[f"{ma_col}_slope_5"] = data[ma_col].pct_change(5)
        data[f"{ma_col}_slope_10"] = data[ma_col].pct_change(10)
        data[f"{ma_col}_slope_25"] = data[ma_col].pct_change(25)

        # 二阶导数（加速度）
        # Second derivative (acceleration)
        data[f"{ma_col}_acc"] = data[ma_col].diff().diff()

        # 角度斜率：均线变化的角度
        # Angle slope: Angle of moving average change
        data[f"ma_{n}_angle"] = np.arctan(
            data[ma_col].diff() / (data[ma_col].shift(1).abs() + 1e-9)
        )

    # =========================
    # 3. Volume factors
    # =========================

    # 1. Volume returns: 计算成交量的日收益率
    # Volume returns: Calculate daily returns of trading volume
    data["vol_ret_1"] = data["volume"].pct_change()
    data["vol_ret_3"] = data["volume"].pct_change(3)
    data["vol_ret_5"] = data["volume"].pct_change(5)
    data["vol_ret_10"] = data["volume"].pct_change(10)
    data["vol_ret_20"] = data["volume"].pct_change(20)

    # 2. Volume MA & bias: 成交量的滚动均值和偏离度
    # Volume MA & bias: Rolling mean of volume and its bias
    for n in vol_ma_windows:
        ma_col = f"vol_ma_{n}"
        data[ma_col] = data["volume"].rolling(n).mean()
        data[f"vol_ma_bias_{n}"] = data["volume"] / data[ma_col] - 1
    
    # 3. Volume-Holding MA & Bias: 成交量与持仓量的偏离度
    # Volume-Holding MA & Bias: Bias between volume and open interest
    for n in vol_ma_windows:
        ma_col = f"vol_hold_ma_{n}"
        data[ma_col] = data["hold"].rolling(n).mean()  
        data[f"vol_hold_ma_bias_{n}"] = data["volume"] / data[ma_col] - 1 

    # 4. Volume volatility: 成交量波动性（滚动标准差）
    # Volume volatility: Rolling standard deviation of volume returns
    for n in vol_ma_windows:
        data[f"vol_vol_{n}"] = data["volume"].pct_change().rolling(n).std()

    # 5. Volume–price divergence: 成交量与价格的背离
    # Volume–price divergence: Divergence between volume and price
    data["vol_price_div_1"] = data["volume"].pct_change() - data["close"].pct_change()
    data["vol_price_div_3"] = data["volume"].pct_change(3) - data["close"].pct_change(3)
    data["vol_price_div_5"] = data["volume"].pct_change(5) - data["close"].pct_change(5)

    # =========================
    # 4. Hold (open interest) factors
    # =========================

    # 1. Hold returns: 计算持仓量的日收益率
    # Hold returns: Calculate daily returns of open interest
    data["hold_ret_1"] = data["hold"].pct_change()
    data["hold_ret_3"] = data["hold"].pct_change(3)
    data["hold_ret_5"] = data["hold"].pct_change(5)
    data["hold_ret_10"] = data["hold"].pct_change(10)
    data["hold_ret_20"] = data["hold"].pct_change(20)

    # 2. Hold MA & bias: 持仓量的滚动均值和偏离度
    # Hold MA & bias: Rolling mean of open interest and its bias
    for n in hold_ma_windows:
        ma_col = f"hold_ma_{n}"
        data[ma_col] = data["hold"].rolling(n).mean()
        data[f"hold_ma_bias_{n}"] = data["hold"] / data[ma_col] - 1

    # 3. Hold volatility: 持仓量波动性（滚动标准差）
    # Hold volatility: Rolling standard deviation of open interest
    for n in hold_ma_windows:
        data[f"hold_vol_{n}"] = data["hold"].pct_change().rolling(n).std()

    # 4. Price–OI divergence: 价格与持仓量之间的背离
    # Price–OI divergence: Divergence between price and open interest
    data["price_oi_div_1"] = data["close"].pct_change() - data["hold"].pct_change()
    data["price_oi_div_3"] = data["close"].pct_change(3) - data["hold"].pct_change(3)
    data["price_oi_div_5"] = data["close"].pct_change(5) - data["hold"].pct_change(5)

    # 5. Volume / hold ratio: 成交量与持仓量的比率
    # Volume / hold ratio: Ratio of volume to open interest
    data["volume_hold_ratio"] = data["volume"] / (data["hold"] + 1e-6)

    # 6. Volume–price & Hold_price correlation: 成交量与价格、持仓量与价格的相关性
    # Volume–price & Hold_price correlation: Correlation between volume and price, and between open interest and price
    data["vp_ratio_5"] = (
        data["volume"].pct_change().rolling(5).corr(data["close"].pct_change())
    )
    data["vp_ratio_10"] = (
        data["volume"].pct_change().rolling(10).corr(data["close"].pct_change())
    )
    data["hp_ratio_5"] = (
        data["hold"].pct_change().rolling(5).corr(data["close"].pct_change())
    )
    data["hp_ratio_10"] = (
        data["hold"].pct_change().rolling(10).corr(data["close"].pct_change())
    )

    # =========================
    # 5. Combined price–volume–OI factors
    # =========================
    data["price_vol_confirm"] = np.sign(data["ret_1"]) * data["vol_ret_1"]
    data["price_oi_trend"] = np.sign(data["ret_1"]) * data["hold_ret_1"]
    data["absorption_ratio"] = data["volume"] / (data["body"].abs() + 1e-3)
    data["effort_result_body_abs"] = data["volume"] * data["body"].abs()
    data["pv_corr_5"] = data["ret_1"].rolling(5).corr(data["vol_ret_1"])
    data["po_corr_5"] = data["ret_1"].rolling(5).corr(data["hold_ret_1"])
    data["vo_corr_5"] = data["vol_ret_1"].rolling(5).corr(data["hold_ret_1"])

    # RSI: 相对强弱指数，衡量市场超买或超卖
    # RSI: Relative Strength Index, measures market overbought or oversold conditions
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator: 随机振荡器，识别超买或超卖状态
    # Stochastic Oscillator: Identifies overbought or oversold conditions
    data["stochastic_14"] = (data["close"] - data["low"].rolling(14).min()) / (data["high"].rolling(14).max() - data["low"].rolling(14).min()) * 100

    return data

