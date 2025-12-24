import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import kurtosis

__all__ = ["check_missing_values", "fill_missing_values", "FeaturePreprocessor"]


# ============================
# 1. 缺失值检验
# ============================
def check_missing_values(data: pd.DataFrame, valid_by_finite: bool = False) -> pd.DataFrame:
    """
    遍历所有列（因子），统计 NaN 和 inf（+inf/-inf）的分布，并把“缺失/异常”拆成：
      1) 总 NaN / 总 Inf
      2) 有效区间内 NaN / Inf
      3) 有效区间外（leading + trailing）NaN / Inf

    有效区间的定义：
      - valid_by_finite=False（默认）：沿用原逻辑，用“非 NaN”的首/尾作为有效区间边界。
        也就是说：inf 不会影响有效区间边界，只影响区间内/外的 Inf 计数。
      - valid_by_finite=True：用“finite（既非 NaN 也非 inf）”的首/尾作为有效区间边界。
        这对你目前存在 inf 的场景更严格、更合理。

    返回：
      每个因子一行的统计表 DataFrame。
    """
    missing_summary = [] 

    for col in data.columns:
        s = data[col] 

        # ========= 1) 构建三种 mask（布尔数组） =========
        is_nan = s.isna() 
        is_inf = np.isinf(s.to_numpy(dtype=float, copy=False)) 
        is_nonfinite = is_nan.to_numpy() | is_inf

        # ========= 2) 统计全列的 NaN / Inf 数量 =========
        total_nan = int(is_nan.sum())
        total_inf = int(is_inf.sum())

        # ========= 3) 决定“有效区间”的首尾边界 =========
        if valid_by_finite:
            # finite_mask=True 表示该位置是有效数（既不是 NaN，也不是 inf）
            finite_mask = ~is_nonfinite

            if finite_mask.any():
                finite_arr = finite_mask.to_numpy()

                # np.argmax 找到第一个 True 的位置（第一个有效数）
                first_pos = int(np.argmax(finite_arr))
                # np.where 找到所有 True 的位置，取最后一个就是最后一个有效数
                last_pos = int(np.where(finite_arr)[0][-1])

                first_valid_index = s.index[first_pos]
                last_valid_index = s.index[last_pos]
            else:
                # 全列都 non-finite（全 NaN/inf）：没有有效区间
                first_valid_index = None
                last_valid_index = None

        else:
            # 默认逻辑：只按“非 NaN”的首尾决定有效区间（inf 会被当成有效值）
            first_valid_index = s.first_valid_index()
            last_valid_index = s.last_valid_index()

        # ========= 4) 统计有效区间内/外的 NaN / Inf =========
        if (first_valid_index is not None) and (last_valid_index is not None):
            # 取有效区间（包含首尾）
            in_range = s.loc[first_valid_index:last_valid_index]

            # 区间内 NaN 数
            in_nan = int(in_range.isna().sum())
            # 区间内 Inf 数
            in_inf = int(np.isinf(in_range.to_numpy(dtype=float, copy=False)).sum())

            # 区间外（leading + trailing）= 总量 - 区间内
            out_nan = total_nan - in_nan
            out_inf = total_inf - in_inf
        else:
            # 没有有效区间：区间内都算 0，区间外就是总量
            in_nan = in_inf = 0
            out_nan, out_inf = total_nan, total_inf

        # ========= 5) 汇总本列统计结果 =========
        missing_summary.append(
            {
                "Factor": col,

                # 全列统计
                "Total NaN": total_nan,
                "Total Inf": total_inf,

                # 有效区间内统计
                "NaN in Valid Range": in_nan,
                "Inf in Valid Range": in_inf,

                # 有效区间外统计（leading + trailing）
                "NaN outside Valid Range": out_nan,
                "Inf outside Valid Range": out_inf,

                # 有效区间边界（方便你定位“哪天开始有效/哪天结束有效”）
                "First Valid Index": first_valid_index,
                "Last Valid Index": last_valid_index,
            }
        )

    # list[dict] -> DataFrame
    missing_summary_df = pd.DataFrame(missing_summary)

    # ========= 6) 设置显示选项（保证输出表不会被截断） =========
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    return missing_summary_df


# ===============================
# 2. 缺失值处理
# ===============================
def fill_missing_values(data: pd.DataFrame):
    data = data.copy()

    # 获取每个因子的有效区间
    def get_valid_range(col):
        first_valid_index = data[col].first_valid_index()
        last_valid_index = data[col].last_valid_index()
        if pd.notna(first_valid_index) and pd.notna(last_valid_index):
            return data.loc[first_valid_index:last_valid_index, col]
        else:
            return data[col]  # 如果没有有效范围，返回整个列

    # ========== 4. 检验后, 特殊处理部分 ==========

    # 1) 这些 slope 类因子：用“前两天均值”填补 NaN
    special_prev2_cols = [
        "vol_2_slope", "vol_2_slope_3", "vol_2_slope_5", "vol_2_slope_10", "vol_2_slope_25",
        "mom_16_slope", "mom_16_slope_3", "mom_16_slope_5", "mom_16_slope_10", "mom_16_slope_25",
        "mom_8_slope", "mom_8_slope_3", "mom_8_slope_5", "mom_8_slope_10", "mom_8_slope_25",
        "mom_4_slope", "mom_4_slope_3", "mom_4_slope_5", "mom_4_slope_10", "mom_4_slope_25",
        "mom_2_slope", "mom_2_slope_3", "mom_2_slope_5", "mom_2_slope_10", "mom_2_slope_25",
    ]

    for c in special_prev2_cols:
        if c not in data.columns:
            continue
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before prev2-mean filling {c}, NaN count: {data[c].isna().sum()}")
            prev2_mean = valid_range.shift(1).rolling(window=2, min_periods=1).mean()
            data.loc[valid_range.index, c] = data.loc[valid_range.index, c].fillna(prev2_mean)
            print(f"After  prev2-mean filling {c}, NaN count: {data[c].isna().sum()}")

    # 2) `mom_2_slope_3` 和 `mom_4_slope_3`：前值填补
    for c in ["mom_2_slope_3", "mom_4_slope_3"]:
        if c not in data.columns:
            continue
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill")
            print(f"After  filling {c}, NaN count: {data[c].isna().sum()}")

    # 3) 比率因子：前值填补
    for c in ["body_pct", "upper_shadow_pct", "lower_shadow_pct"]:
        if c not in data.columns:
            continue
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill")
            print(f"After  filling {c}, NaN count: {data[c].isna().sum()}")

    # 4) 删除所有因子中“最晚开始有有效值”之前的行（确保所有因子都有有效值）
    first_pos = []
    for c in data.columns:
        m = data[c].notna().to_numpy()
        if m.any():
            first_pos.append(int(m.argmax()))
    max_leading = max(first_pos) if first_pos else 0

    return data.iloc[max_leading:].copy()


# =============================
# 3. 因子预处理
# =============================
class FeaturePreprocessor:
    def __init__(
        self,
        heavy_kurtosis=6,              # 重尾判定的峰度阈值（kurtosis > heavy_kurtosis）
        clip_pct=(0.1, 99.9),          # 重尾列的分位数裁剪区间
        log_candidates=("volume", "hold"),  # 强制按重尾处理的候选列
        sigma_clip=6,                 # 全因子 σ 裁剪倍数（均值±sigma_clip*std）
        robust=False,                 # True 用 RobustScaler，否则 StandardScaler
        check_finite=True,            # True 则强制检查 NaN/inf
    ):
        self.heavy_kurtosis = heavy_kurtosis
        self.clip_low, self.clip_high = clip_pct
        self.log_candidates = tuple(log_candidates)
        self.sigma_clip = sigma_clip
        self.use_robust = robust
        self.check_finite = check_finite

        self.feature_cols = None      # 因子列名列表
        self.heavy_cols = []          # 重尾列名列表

        self.clip_bounds = {}         # 重尾列分位数边界：{col: (lo, hi)}
        self.sigma_bounds = {}        # σ裁剪统计：{col: (mean, std)}

        self.scaler = None            # 标准化器（fit 在 train 上）
        self.is_fitted = False        # 是否已 fit

    # ============================
    # 工具：检查 NaN/inf
    # ============================
    def _check_no_nan_inf(self, X: pd.DataFrame, where: str):
        if not self.check_finite:
            return

        arr = X.to_numpy(dtype=float, copy=False)
        if np.isfinite(arr).all():
            return

        bad = ~np.isfinite(arr)
        i, j = np.argwhere(bad)[0]
        col = X.columns[j]
        idx = X.index[i]
        v = X.iloc[i, j]

        raise ValueError(
            f"[{where}] 发现非有限值（NaN/inf）：index={idx}, col='{col}', value={v}. "
            "本预处理器不负责处理 NaN/inf，请先在外部清洗。"
        )

    # ============================
    # 1) 自动检测重尾列
    # ============================
    def _detect_heavy(self, X: pd.DataFrame):
        heavy = []
        for c in self.feature_cols:
            if ("slope" in c) or ("ret" in c):
                continue

            x = X[c].to_numpy(dtype=float, copy=False)
            try:
                k = kurtosis(x, fisher=False, bias=False)
                if np.isfinite(k) and (k > self.heavy_kurtosis):
                    heavy.append(c)
            except Exception:
                # 检测失败则不纳入 heavy
                pass

        return heavy

    # ============================
    # 2) 重尾处理：分位数裁剪 + signed-log
    # ============================
    def _signed_log_clip(self, x: np.ndarray, col: str, fit: bool):
        x = np.asarray(x, dtype=float)

        if fit:
            lo, hi = np.percentile(x, [self.clip_low, self.clip_high])

            # 常数/异常列兜底：避免 lo>=hi
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo >= hi):
                m = float(np.mean(x))
                eps = 1e-12
                lo, hi = m - eps, m + eps

            self.clip_bounds[col] = (float(lo), float(hi))
        else:
            lo, hi = self.clip_bounds[col]

        x = np.clip(x, lo, hi)
        return np.sign(x) * np.log1p(np.abs(x))

    # ============================
    # 3) σ裁剪：均值±k*std（所有列）
    # ============================
    def _sigma_clip(self, x: np.ndarray, col: str, fit: bool):
        x = np.asarray(x, dtype=float)

        if fit:
            m = float(np.mean(x))
            sd = float(np.std(x)) + 1e-12
            self.sigma_bounds[col] = (m, sd)
        else:
            m, sd = self.sigma_bounds[col]

        lo = m - self.sigma_clip * sd
        hi = m + self.sigma_clip * sd
        return np.clip(x, lo, hi)

    # ============================
    # 4) 训练期拟合：只用 train 学参数
    # ============================
    def fit(self, df: pd.DataFrame, feature_cols):
        """
        前提：df[feature_cols] 已在外部清洗完 NaN/inf。
        流程：
        1) 识别 heavy 列（自动 + 手动）
        2) heavy：分位数裁剪 + signed-log（记录 clip_bounds）
        3) 全列：σ裁剪（记录 mean/std）
        4) 标准化器 fit（仅 train）
        """
        assert df.index.is_monotonic_increasing, "索引必须按时间升序"
        assert df.index.is_unique, "索引必须唯一"
        assert isinstance(df.index, pd.DatetimeIndex), "索引必须是 DatetimeIndex"

        self.feature_cols = list(feature_cols)
        X = df[self.feature_cols].copy()

        self._check_no_nan_inf(X, where="fit")

        auto_heavy = self._detect_heavy(X)
        manual_heavy = [c for c in self.log_candidates if c in self.feature_cols]
        self.heavy_cols = sorted(set(auto_heavy + manual_heavy))

        Xp = X.copy()

        for col in self.heavy_cols:
            Xp[col] = self._signed_log_clip(Xp[col].to_numpy(copy=False), col, fit=True)

        for col in self.feature_cols:
            Xp[col] = self._sigma_clip(Xp[col].to_numpy(copy=False), col, fit=True)

        self.scaler = RobustScaler() if self.use_robust else StandardScaler()
        self.scaler.fit(Xp[self.feature_cols])

        self.is_fitted = True
        return self

    # ============================
    # 5) 预测期变换：只用 train 参数做映射
    # ============================
    def transform(self, df: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("请先 fit(train)，再 transform(test/val)。")

        assert df.index.is_monotonic_increasing, "索引必须按时间升序"
        assert df.index.is_unique, "索引必须唯一"
        assert isinstance(df.index, pd.DatetimeIndex), "索引必须是 DatetimeIndex"

        X = df[self.feature_cols].copy()
        self._check_no_nan_inf(X, where="transform")

        Xp = X.copy()

        for col in self.heavy_cols:
            Xp[col] = self._signed_log_clip(Xp[col].to_numpy(copy=False), col, fit=False)

        for col in self.feature_cols:
            Xp[col] = self._sigma_clip(Xp[col].to_numpy(copy=False), col, fit=False)

        Z = self.scaler.transform(Xp[self.feature_cols])

        out = df.copy()
        out[self.feature_cols] = Z
        return out

    # ============================
    # 简单统计报告
    # ============================
    def report(self, df: pd.DataFrame):
        X = df[self.feature_cols]
        stats = X.describe().T
        print(stats[["mean", "std", "min", "max"]])
        print("\nMax abs:", X.abs().to_numpy().max())


