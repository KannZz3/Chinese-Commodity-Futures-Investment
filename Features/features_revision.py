import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import kurtosis

__all__ = ["check_missing_values", "fill_missing_values", "FeaturePreprocessor"]


# ============================
# 1. 缺失值检验
# ============================
def check_missing_values(data: pd.DataFrame):
    """
    遍历所有列（因子），返回每个因子的总缺失值数量以及有效范围内的缺失值数量。
    参数:
    data : pd.DataFrame
        含有因子的 DataFrame
    返回:
    missing_summary : pd.DataFrame
        包含每个因子的总缺失值数量、有效范围内缺失值数量等信息
    """
    missing_summary = []
    for col in data.columns:
        total_missing = data[col].isna().sum()
        
        # 找到第一个有效值和最后一个有效值的索引
        first_valid_index = data[col].first_valid_index()
        last_valid_index = data[col].last_valid_index()

        if pd.notna(first_valid_index) and pd.notna(last_valid_index):
            valid_range = data.loc[first_valid_index:last_valid_index, col]
            
            missing_in_valid_range = valid_range.isna().sum()
        else:
            missing_in_valid_range = 0  

        # 记录结果
        missing_summary.append({
            'Factor': col,
            'Total Missing Values': total_missing,
            'Missing Values in Valid Range': missing_in_valid_range
        })

    missing_summary_df = pd.DataFrame(missing_summary)
    
    # 设置 pandas 显示选项，确保显示所有行和列
    pd.set_option('display.max_rows', None) 
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', None) 
    pd.set_option('display.max_colwidth', None) 

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

    # ========== 1. 动量 / slope / ret类 ==========
    for c in data.columns:
        if c.startswith("mom_") or "slope" in c or c == "ret_1":
            valid_range = get_valid_range(c)
            if valid_range.size > 0:  # 只在有效范围内填补
                print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
                data[c] = data[c].fillna(method='ffill')  # 前值填补
                print(f"After filling {c}, NaN count: {data[c].isna().sum()}")
               
    # ========== 2. MA 均线 ==========
    for c in data.columns:
        if c.startswith("ma_") and "slope" not in c:
            valid_range = get_valid_range(c)
            if valid_range.size > 0:  # 只在有效范围内填补
                print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
                data[c] = data[c].fillna(data["close"])
                print(f"After filling {c}, NaN count: {data[c].isna().sum()}")

    # ========== 4. 检验后, 特殊处理部分 ==========
    # 处理 `mom_2_slope_3` 和 `mom_4_slope_3` 的缺失值，使用前值填补
    for c in ["mom_2_slope_3", "mom_4_slope_3"]:
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill")  
            print(f"After filling {c}, NaN count: {data[c].isna().sum()}")

    # 处理比率因子的缺失值：使用前值填补
    for c in ["body_pct", "upper_shadow_pct", "lower_shadow_pct"]:
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill") 
            print(f"After filling {c}, NaN count: {data[c].isna().sum()}")
    first_pos = []
    for c in data.columns:
        m = data[c].notna().to_numpy()
        if m.any():                      # 该列至少有一个有效值
            first_pos.append(int(m.argmax()))  # 首个有效值位置 = leading NaN 行数
    max_leading = max(first_pos) if first_pos else 0

    return data.iloc[max_leading:].copy()


# =========================
# 3. 因子预处理
# =========================

