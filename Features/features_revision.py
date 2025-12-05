import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import kurtosis

__all__ = ["fill_missing_values", "FeaturePreprocessor"]


# =========================
# 1. 缺失值填补
# =========================
def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    按因子类型对缺失值进行合理填补。

    参数
    ----
    data : pd.DataFrame
        含有各种因子的 DataFrame

    返回
    ----
    filled : pd.DataFrame
        填补缺失值后的 DataFrame
    """
    data = data.copy()

    # ========== 1. 动量 / slope / ret 类 ==========
    for c in data.columns:
        if c.startswith("mom_") or "slope" in c or c == "ret_1":
            data[c] = data[c].fillna(0)

    # ========== 2. 波动率类 ==========
    for c in data.columns:
        # 包含 vol_ret、vol_ma、vol_vol 等
        if c.startswith("vol_"):
            data[c] = data[c].fillna(0)

    # ========== 3. MA 均线 ==========
    for c in data.columns:
        if c.startswith("ma_") and "slope" not in c:
            # 均线缺失时，用当前 close 替代，避免开头大量 NaN
            data[c] = data[c].fillna(data["close"])

    # ========== 4. 均线乖离 ==========
    for c in data.columns:
        if c.startswith("close_ma_"):
            data[c] = data[c].fillna(0)

    # ========== 5. Hold / Volume 系 ==========
    for c in data.columns:
        if (
            c.startswith("vol_")
            or c.startswith("hold_")
            or c.startswith("price_oi_")
            or c.startswith("vol_price_")
            or c == "volume_hold_ratio"
            or c == "vp_ratio"
        ):
            data[c] = data[c].fillna(0)

    # ========== 6. 最后兜底 ==========
    data = data.fillna(0)

    return data


# ============================================
# 2. 预处理（Robust Feature Preprocessor）
# ============================================
class FeaturePreprocessor:
    """
    交易安全版特征预处理：
      - 重尾检测 + signed-log
      - slope σ-clip
      - 全局 σ-clip
      - NaN 安全处理（仅用历史信息）
      - 标准化（仅用 train 部分 fit）

    用法:
        fp = FeaturePreprocessor(robust=True)
        fp.fit(train_df, feature_cols)
        train_proc = fp.transform(train_df)
        test_proc  = fp.transform(test_df)
    """

    def __init__(
        self,
        heavy_kurtosis=6,
        clip_pct=(1, 99),
        log_candidates=("volume", "hold"),
        slope_sigma=8,
        robust=False,
        global_sigma=5,
    ):
        self.heavy_kurtosis = heavy_kurtosis
        self.clip_low, self.clip_high = clip_pct
        self.log_candidates = log_candidates
        self.slope_sigma = slope_sigma
        self.global_sigma = global_sigma
        self.use_robust = robust

        # training stats
        self.feature_cols = None
        self.heavy_cols = []
        self.clip_bounds = {}
        self.slope_bounds = {}
        self.global_mean = {}
        self.global_std = {}
        self.medians = {}
        self.scaler = None
        self.is_fitted = False

    # ============================
    # 1) detect heavy-tail columns
    # ============================
    def _detect_heavy(self, df: pd.DataFrame):
        heavy = []
        for c in self.feature_cols:
            if "slope" in c:
                continue
            if "ret" in c:
                continue
            try:
                if kurtosis(df[c].dropna(), fisher=False) > self.heavy_kurtosis:
                    heavy.append(c)
            except Exception:
                pass
        return heavy

    # ============================
    # 2) signed-log with safe clip
    # ============================
    def _signed_log_clip(self, x, col, fit: bool):
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)

        # 先 ffill 保持时间结构
        s = pd.Series(x).ffill().values

        # 用训练时全局中位数填剩余 nan（不会泄漏未来）
        if fit:
            med = np.nanmedian(s)
        else:
            med = self.medians[col]
        s = np.nan_to_num(s, nan=med)

        if fit:
            lo, hi = np.nanpercentile(s, [self.clip_low, self.clip_high])

            # fallback 避免 lo/hi 非法
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or lo >= hi:
                lo = np.nanpercentile(s, 10)
                hi = np.nanpercentile(s, 90)

            self.clip_bounds[col] = (lo, hi)
        else:
            lo, hi = self.clip_bounds[col]

        s = np.clip(s, lo, hi)
        return np.sign(s) * np.log1p(np.abs(s))

    # ============================
    # 3) slope σ-clip
    # ============================
    def _clip_slope(self, x, col, fit: bool):
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)

        # ffill + median
        s = pd.Series(x).ffill().values
        med = np.nanmedian(s)
        s = np.nan_to_num(s, nan=med)

        if fit:
            m = np.nanmean(s)
            sd = np.nanstd(s) + 1e-6
            self.slope_bounds[col] = (m, sd)
        else:
            m, sd = self.slope_bounds[col]

        return np.clip(
            s,
            m - self.slope_sigma * sd,
            m + self.slope_sigma * sd,
        )

    # ============================
    # 4) fit(train)
    # ============================
    def fit(self, df: pd.DataFrame, feature_cols):
        assert df.index.is_monotonic_increasing, "df 必须按时间升序排列"

        self.feature_cols = list(feature_cols)

        # --- step 1: 全局 inf→nan ---
        df_proc = df.copy().replace([np.inf, -np.inf], np.nan)

        # --- step 2: ffill 修复为“历史唯一来源” ---
        df_proc = df_proc.ffill()

        # --- step 3: 用 train 的 median 填补剩余 nan（无未来泄漏） ---
        self.medians = df_proc[self.feature_cols].median().to_dict()
        df_proc = df_proc.fillna(self.medians)

        # --- heavy-tail detection ---
        auto_heavy = self._detect_heavy(df_proc)
        manual_heavy = [c for c in self.log_candidates if c in self.feature_cols]
        self.heavy_cols = sorted(set(auto_heavy + manual_heavy))

        # --- heavy-tail 修正 ---
        for col in self.heavy_cols:
            df_proc[col] = self._signed_log_clip(df_proc[col], col, fit=True)

        # --- slope σ-clip ---
        for col in self.feature_cols:
            if "slope" in col:
                df_proc[col] = self._clip_slope(df_proc[col], col, fit=True)

        # --- global σ-clip（基于 train） ---
        for col in self.feature_cols:
            m = df_proc[col].mean()
            sd = df_proc[col].std() + 1e-6
            self.global_mean[col] = m
            self.global_std[col] = sd
            df_proc[col] = df_proc[col].clip(
                m - self.global_sigma * sd,
                m + self.global_sigma * sd,
            )

        # --- 标准化器（fit on train only） ---
        self.scaler = RobustScaler() if self.use_robust else StandardScaler()
        self.scaler.fit(df_proc[self.feature_cols])

        self.is_fitted = True
        return self

    # ============================
    # 5) transform(test / val / live)
    # ============================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("请先 fit(train) 再 transform(test)")

        df2 = df.copy().replace([np.inf, -np.inf], np.nan)

        # ffill（保持时间结构）但不能用未来数据
        df2 = df2.ffill()

        # nan → 训练集的 median（不会泄漏未来）
        for col in self.feature_cols:
            df2[col] = df2[col].fillna(self.medians[col])

        # heavy-tail 修正（基于 train bounds）
        for col in self.heavy_cols:
            df2[col] = self._signed_log_clip(df2[col], col, fit=False)

        # slope σ-clip（基于 train bounds）
        for col in self.feature_cols:
            if "slope" in col:
                df2[col] = self._clip_slope(df2[col], col, fit=False)

        # global σ-clip（基于 train stats）
        for col in self.feature_cols:
            m = self.global_mean[col]
            sd = self.global_std[col]
            df2[col] = df2[col].clip(
                m - self.global_sigma * sd,
                m + self.global_sigma * sd,
            )

        # 标准化（基于 train）
        df2[self.feature_cols] = self.scaler.transform(df2[self.feature_cols])

        return df2

    # ============================
    # 6) 简单报告
    # ============================
    def report(self, df: pd.DataFrame):
        """
        打印当前特征的简单统计，方便 sanity check。
        """
        stats = df[self.feature_cols].describe().T
        print(stats[["mean", "std", "min", "max"]])
        print("\nMax abs:", df[self.feature_cols].abs().max().max())
        return stats
