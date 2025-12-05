
import numpy as np
import pandas as pd

__all__ = [
    "choose_base_th_from_train",
    "choose_T_from_train",
    "cta_backtest_money",
]


def choose_base_th_from_train(
    train_df,
    close_col="close",
    base_th_ref=0.01,     # 在日波动 ≈ ref_vol 时，目标 cum_ret 阈值（默认 1%）
    vol_lookback=50,      # 滚动波动率窗口
    ref_vol=0.01,         # 参考“基准波动率”，默认 1%
    floor_mult=0.5,       # base_th 不低于 base_th_ref * floor_mult
    ceil_mult=3.0,        # base_th 不高于 base_th_ref * ceil_mult
    winsor_pct=1.0,       # ⭐ 对日收益两端各 winsor_pct% 的截断（例如 1%）
    min_vol=0.002,        # ⭐ vol_median 下限（例如 0.2%）
    max_vol=0.05,         # ⭐ vol_median 上限（例如 5%）
    verbose=True
):
    """
    根据训练集价格数据，选择一个合适的 base_th（单笔交易目标 cum_ret 阈值）。
    """
    # 1) 日收益
    if close_col not in train_df.columns:
        raise KeyError(f"[choose_base_th_from_train] '{close_col}' 不在 train_df 列中。")

    close = train_df[close_col].astype(float)
    ret = close.pct_change().dropna()

    if len(ret) < vol_lookback:
        if verbose:
            print("[choose_base_th_from_train] 数据太少，直接返回 base_th_ref。")
        return float(base_th_ref)

    # 1.5) ⭐ 对日收益做轻微 winsor（heavy-tail 友好）
    if winsor_pct is not None and winsor_pct > 0:
        # winsor_pct = 1 → 1% & 99% 分位数
        low_q, high_q = np.percentile(ret, [winsor_pct, 100 - winsor_pct])
        ret = ret.clip(low_q, high_q)

    # 2) 训练集的滚动波动率中位数
    rolling_vol = ret.rolling(vol_lookback).std()
    vol_median_raw = rolling_vol.median()

    if (vol_median_raw is None) or (not np.isfinite(vol_median_raw)) or (vol_median_raw <= 0):
        if verbose:
            print("[choose_base_th_from_train] vol_median 无效，直接返回 base_th_ref。")
        return float(base_th_ref)

    vol_median = float(vol_median_raw)

    # 2.5) ⭐ 给 vol_median 加一个“合理区间”防护
    vol_median_before_clip = vol_median

    if min_vol is not None:
        vol_median = max(vol_median, float(min_vol))
    if max_vol is not None:
        vol_median = min(vol_median, float(max_vol))

    # 3) 按波动缩放的 data-driven base_th
    base_th_data = base_th_ref * (vol_median / ref_vol)

    # 4) floor / ceil
    base_th_floor = floor_mult * base_th_ref
    base_th_ceil = ceil_mult * base_th_ref

    base_th_clipped = max(base_th_data, base_th_floor)
    base_th_clipped = min(base_th_clipped, base_th_ceil)

    if verbose:
        print(
            "[choose_base_th_from_train] "
            f"vol_median_raw={vol_median_before_clip:.4f}, "
            f"vol_median_used={vol_median:.4f}, "
            f"base_th_data={base_th_data:.4f}, "
            f"floor={base_th_floor:.4f}, ceil={base_th_ceil:.4f} "
            f"→ base_th={base_th_clipped:.4f}"
        )

    return float(base_th_clipped)


def choose_T_from_train(
    train_df,
    score_col="score",
    q=0.80,          # 默认用 |score| 的 80% 分位数（比 0.95 宽松很多）
    min_T=0.5,       # ⭐ 你的直觉：至少要 > 0.7 才出手
    max_T=None,      # 可选：给一个上限，避免阈值太大
    verbose=True
):
    """
    根据训练集 score 分布，选择一个 CTA 信号阈值 T（用于 |score| > T 才开仓）。
    """
    if score_col not in train_df.columns:
        raise KeyError(f"[choose_T_from_train] '{score_col}' 不在 train_df 列中。")

    s = train_df[score_col].dropna().astype(float)
    if len(s) == 0:
        raise ValueError("[choose_T_from_train] 训练集 score 全为 NaN，无法估计 T。")

    # 1) data-driven 阈值：q 分位数
    T_data = s.abs().quantile(q)

    # 2) 至少不小于 min_T（例如 0.7）
    T_final = max(T_data, float(min_T))

    # 3) 可选的上限
    if max_T is not None:
        T_final = min(T_final, float(max_T))

    if verbose:
        msg = (
            f"[choose_T_from_train] "
            f"T_data(q={q:.2f})={T_data:.4f}, "
            f"min_T={min_T:.4f}"
        )
        if max_T is not None:
            msg += f", max_T={max_T:.4f}"
        msg += f" → T={T_final:.4f}"
        print(msg)

    return float(T_final)


def cta_backtest_money(
    df,
    score_col="score",
    T=None,                 # ⭐ 必须由外部传入（例如 choose_T_from_train 的输出）
    base_th=0.01,           # 建议传入 choose_base_th_from_train 的输出
    max_hold_days=8,        # 最多持有 8 日
    init_capital=500000,    # 初始资金 50w
    pos_pct=0.3,            # 每次用 30% 净值建仓
    leverage=10,            # 方向杠杆
    shift_signal=True       # 是否下一根K线再执行信号
):
    """
    高级 CTA 回测（真实资金版，整理版）
    """
    df = df.copy()
    # === 0) 标的日收益 ===
    df["ret"] = df["close"].pct_change().fillna(0.0)

    if T is None:
        raise ValueError(
            "[cta_backtest_money] 参数 T 未指定。"
            "请先在训练集上用 choose_T_from_train 等函数计算 T，然后传入本函数。"
        )

    raw_score = df[score_col]

    # score → 原始方向信号：+1 / -1 / 0
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[raw_score >  T] =  1.0
    sig[raw_score < -T] = -1.0

    if shift_signal:
        # 信号下一日生效
        sig = sig.shift(1).fillna(0.0)

    df["signal_raw"] = sig

    # === 2) 初始化策略状态 ===
    capital = init_capital      # 账户净值
    pos_dir = 0.0               # 纯方向：+1 / -1 / 0
    hold_value = 0.0            # 名义持仓资金（不含杠杆倍数）
    cum_ret = 0.0               # 本笔交易累计方向化收益（含 leverage）
    hold_days = 0               # 持仓天数
    reached_tp1 = False         # 是否触发过 base_th

    capital_list = []
    pos_exposure_list = []      # 记录方向敞口：pos_dir * leverage
    hold_value_list = []

    # === 3) 主循环：逐日更新 ===
    for i in range(len(df)):
        daily_ret_raw = df["ret"].iloc[i]
        signal_today = df["signal_raw"].iloc[i]

        # --- A. 空仓 → 看看要不要开仓 ---
        if pos_dir == 0.0 and signal_today != 0.0:
            pos_dir = signal_today               # 方向：+1 / -1
            hold_value = capital * pos_pct       # 用当前净值的一部分做名义资金
            cum_ret = 0.0
            hold_days = 0
            reached_tp1 = False

        # --- B. 当日收益（真实资金） ---
        effective_notional = pos_dir * leverage * hold_value
        daily_profit = effective_notional * daily_ret_raw

        # 更新账户净值
        capital += daily_profit

        # --- C. 若在持仓中，更新本笔交易指标 ---
        if pos_dir != 0.0:
            cum_ret += pos_dir * leverage * daily_ret_raw
            hold_days += 1

            if cum_ret >= base_th:
                reached_tp1 = True

            # --- D. 退出条件 ---
            exit_flag = False
            if cum_ret >= 2 * base_th:                 # 强止盈
                exit_flag = True
            elif cum_ret <= -0.9 * base_th:            # 止损
                exit_flag = True
            elif reached_tp1 and cum_ret <= 0.6 * base_th:  # 回撤止盈
                exit_flag = True
            elif hold_days >= max_hold_days:           # 时间止损
                exit_flag = True

            if exit_flag:
                pos_dir = 0.0
                hold_value = 0.0
                cum_ret = 0.0
                hold_days = 0
                reached_tp1 = False

        # --- E. 记录每日状态 ---
        capital_list.append(capital)
        pos_exposure_list.append(pos_dir * leverage)  # 0 / ±leverage
        hold_value_list.append(hold_value)

    # === 4) 写回结果列 ===
    df["capital"] = capital_list
    df["pos"] = pos_exposure_list      # 0 / +leverage / -leverage
    df["hold_value"] = hold_value_list
    df["equity"] = df["capital"] / float(init_capital)  # 归一化净值曲线

    return df



