import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report

__all__ = ["train_one_direction_xgb"]


def train_one_direction_xgb(
    df,
    feature_cols,
    target_col,
    direction_name="UP",
    n_estimators=400,
    max_depth=5,
    min_samples_leaf=50,   # -> 映射到 XGB 的 min_child_weight（越大越保守）
    max_features="sqrt",   # -> 映射到 XGB 的 colsample_bytree（每棵树采样的特征比例）

    # ===== OOS split params =====
    valid_size=250,
    n_windows=3,           # 从尾部切 n_windows 个连续 valid block（每块 valid_size）

    # ===== Feature selection params =====
    imp_min=0.0,           # OOS permutation importance 的绝对阈值（可设 0）
    imp_quantile=0.6,      # OOS importance 分位阈值（例如 0.6 表示保留前 40%）
    top_k_max=25,          # 最终最多保留多少个特征
    auc_single_min=0.62,   # 单因子 OOS AUC 均值阈值
    n_perm=5,              # permutation 次数（越大越稳但越慢）
    max_single_check=25,   # 最多对多少个候选特征做“单因子AUC复核”
):
    """
    多窗口时间序列 Walk-Forward（expanding train / forward valid）：
    1) 每个窗口：用 train 拟合 XGB
    2) 在 valid 上做 OOS permutation importance（打乱单个特征列，观察性能下降）
    3) 聚合多个窗口的 OOS importance（mean/median/pos_frac）
    4) 重要性过阈值后，对 top-N 候选特征做“单因子 XGB”的 OOS AUC 复核
    5) 输出最终选中的特征、重要性表、以及最后一个窗口训练出来的 FINAL model

    注意：
    - 该函数内部切分遵循时间顺序：train 永远在 valid 之前（避免训练阶段泄漏）。
    - 但最后输出的 prob_all 是“用最后窗口训练的模型”对全样本打分，适合诊断/画图；
      若要严格回测，请用 walk-forward 的 OOS 预测序列（需要另外实现 oof_prob 填充）。
    """

    print(f"\n========== Start training {direction_name} (XGB), target = {target_col} ==========\n")

    # ==========================================================
    # 0) 时间排序 + 基本合法性检查
    # ==========================================================
    df = df.sort_index()          # 强制按时间升序
    n = len(df)
    if n <= valid_size:
        raise ValueError(f"Need n > valid_size. Got n={n}, valid_size={valid_size}.")

    # 最大可用窗口数：保证最早窗口至少有 1 条 train
    max_w = (n - 1) // valid_size
    if n_windows > max_w:
        print(
            f"[Warn] n_windows={n_windows} too large for n={n}, valid_size={valid_size}. "
            f"Capping to {max_w}."
        )
        n_windows = max_w

    # ==========================================================
    # 1) 从尾部构造 n_windows 个 (train, valid) 窗口
    #    - valid 是连续的一段长度 valid_size
    #    - train 是 valid 之前的全部历史（expanding）
    # ==========================================================
    windows = []
    for w in range(n_windows, 0, -1):
        train_end = n - w * valid_size         # train: [0, train_end)
        valid_start = train_end
        valid_end = train_end + valid_size     # valid: [valid_start, valid_end)
        if train_end <= 0:
            continue
        windows.append((0, train_end, valid_start, valid_end))

    # 这里 windows 已经是从“更早窗口 -> 更晚窗口”（因为 w 从大到小 append）
    # windows = windows  # 这行本身是冗余的，占位说明顺序无需 reverse

    if len(windows) == 0:
        raise RuntimeError("Failed to construct windows. Check n/valid_size/n_windows.")

    print(f"[Info] Using {len(windows)} window(s):")
    for k, (tr0, tr1, va0, va1) in enumerate(windows, 1):
        print(f"  W{k}: train[{tr0}:{tr1}] ({tr1 - tr0}), valid[{va0}:{va1}] ({va1 - va0})")

    # ==========================================================
    # 2) max_features -> colsample_bytree（特征子采样比例）
    # ==========================================================
    d = len(feature_cols)
    if isinstance(max_features, str):
        if max_features == "sqrt":
            # 每棵树采样 sqrt(d) 个特征 -> 比例 sqrt(d)/d
            colsample_bytree = float(np.sqrt(d) / d) if d > 0 else 1.0
        elif max_features == "log2":
            colsample_bytree = float(np.log2(d) / d) if d > 0 else 1.0
        else:
            colsample_bytree = 1.0
    else:
        # 若用户直接给比例（如 0.7），直接用
        colsample_bytree = float(max_features)

    # 特征名 -> 列索引（用于单因子训练时取 [:, [j]]）
    col_to_j = {c: i for i, c in enumerate(feature_cols)}

    # ==========================================================
    # 3) 统一 metric_mode：所有窗口都可算 AUC 才用 AUC，否则用 -logloss
    # ==========================================================
    def has_both_classes(y_arr) -> bool:
        """检查标签是否同时包含 0 和 1（AUC 需要双类）。"""
        return len(np.unique(y_arr)) > 1

    use_auc = True
    for (_, _, va0, va1) in windows:
        y_va = df.iloc[va0:va1][target_col].to_numpy().astype(int)
        if not has_both_classes(y_va):
            use_auc = False
            break

    metric_mode = "auc" if use_auc else "neg_logloss"
    print(f"\n[Info] metric_mode for permutation importance = {metric_mode}")

    def metric_score(y_true, proba_2col):
        """
        统一“越大越好”的分数：
        - auc：直接用 AUC
        - neg_logloss：用 -logloss（取负号使得越大越好）
        """
        if metric_mode == "auc":
            return roc_auc_score(y_true, proba_2col[:, 1])
        else:
            return -log_loss(y_true, proba_2col, labels=[0, 1])

    def build_model(scale_pos_weight: float, random_state: int):
        """构建一个 XGBClassifier（统一超参），并传入类别不平衡权重与随机种子。"""
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=colsample_bytree,
            reg_lambda=1.0,
            min_child_weight=float(min_samples_leaf),  # RF 的 min_samples_leaf -> XGB 的 min_child_weight
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
        )

    # ==========================================================
    # 4) 多窗口：训练模型 + 在 valid 上做 OOS permutation importance
    # ==========================================================
    imp_by_window = []     # 每个窗口一个 Series(perm_importance)
    window_metrics = []    # 每个窗口的 valid 指标（ACC/AUC/baseline等）

    for w, (tr0, tr1, va0, va1) in enumerate(windows, 1):
        # --- 切分 train/valid（按时间顺序）
        train_df = df.iloc[tr0:tr1]
        valid_df = df.iloc[va0:va1]

        # --- 取特征与标签
        X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_tr = train_df[target_col].to_numpy().astype(int)
        X_va = valid_df[feature_cols].to_numpy(dtype=np.float32)
        y_va = valid_df[target_col].to_numpy().astype(int)

        # --- 类别不平衡处理：scale_pos_weight = neg/pos
        pos_ratio_tr = y_tr.mean() if len(y_tr) else np.nan
        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

        # --- 训练该窗口模型（仅用 train）
        model_w = build_model(scale_pos_weight=scale_pos_weight, random_state=2025 + 17 * w)
        model_w.fit(X_tr, y_tr)

        # --- valid 上的 baseline 预测与分数
        proba_va = model_w.predict_proba(X_va)          # shape = (n_valid, 2)
        baseline = metric_score(y_va, proba_va)         # AUC 或 -logloss

        # --- 诊断指标：ACC/AUC（AUC 可能因为单类而 NaN）
        prob_va = proba_va[:, 1]
        pred_va = (prob_va > 0.5).astype(int)
        acc_va = accuracy_score(y_va, pred_va)
        auc_va = roc_auc_score(y_va, prob_va) if has_both_classes(y_va) else np.nan

        # --- 记录窗口表现
        window_metrics.append({
            "window": w,
            "train_len": len(train_df),
            "valid_len": len(valid_df),
            "pos_ratio_train": pos_ratio_tr,
            "acc_valid": acc_va,
            "auc_valid": auc_va,
            "baseline_metric": baseline,
            "metric_mode": metric_mode,
        })

        # --- permutation importance：对每个特征列打乱，观察性能下降
        rng = np.random.RandomState(2025 + 1000 * w)
        perm_importance = {}

        for j, col in enumerate(feature_cols):
            deltas = []
            for _ in range(n_perm):
                # 复制 valid 特征矩阵
                Xp = X_va.copy()

                # 打乱第 j 列（仅在副本上做，不改变原数据）
                shuffled = Xp[:, j].copy()
                rng.shuffle(shuffled)
                Xp[:, j] = shuffled

                # 重新预测 + 重新算分
                proba_p = model_w.predict_proba(Xp)
                metric_p = metric_score(y_va, proba_p)

                # baseline - metric_p = 性能下降量（越大说明该特征越重要）
                deltas.append(baseline - metric_p)

            perm_importance[col] = float(np.mean(deltas))

        # 当前窗口的重要性（从大到小排序）
        imp_ser = pd.Series(perm_importance).sort_values(ascending=False)
        imp_by_window.append(imp_ser)

        # --- 打印窗口结果（便于调参/诊断）
        print(f"\n--- W{w} OOS valid metrics ---")
        print(
            f"train_len={len(train_df)}, valid_len={len(valid_df)}, pos_ratio_train={pos_ratio_tr:.4f}, "
            f"valid_ACC={acc_va:.4f}, valid_AUC={auc_va:.4f}"
        )
        print(f"W{w} OOS permutation importance (Top 10):")
        print(imp_ser.head(10))

    # ==========================================================
    # 4b) 聚合多个窗口的 importance（mean/median/pos_frac）
    # ==========================================================
    imp_mat = pd.concat(imp_by_window, axis=1)   # 行=特征，列=窗口
    imp_mean = imp_mat.mean(axis=1)
    imp_median = imp_mat.median(axis=1)
    imp_pos_frac = (imp_mat > 0).mean(axis=1)    # importance>0 的窗口比例（稳定性）

    importance_series = imp_mean.sort_values(ascending=False)

    print(f"\n[{direction_name}] Aggregated OOS permutation importance (mean) Top 20:")
    top20 = importance_series.head(20).index
    print(pd.DataFrame({
        "imp_mean": imp_mean.loc[top20],
        "imp_median": imp_median.loc[top20],
        "pos_frac": imp_pos_frac.loc[top20],
    }))

    # ==========================================================
    # 5) 用“最后一个窗口”的 train 训练 FINAL 模型，并在其 valid 上做最终 OOS 输出
    # ==========================================================
    tr0, tr1, va0, va1 = windows[-1]
    train_df = df.iloc[tr0:tr1]
    valid_df = df.iloc[va0:va1]

    X_tr = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_tr = train_df[target_col].to_numpy().astype(int)
    X_va = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_va = valid_df[target_col].to_numpy().astype(int)

    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    scale_pos_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

    model = build_model(scale_pos_weight=scale_pos_weight, random_state=2025)
    model.fit(X_tr, y_tr)

    proba_va = model.predict_proba(X_va)
    prob_va = proba_va[:, 1]
    pred_va = (prob_va > 0.5).astype(int)

    acc = accuracy_score(y_va, pred_va)
    auc = roc_auc_score(y_va, prob_va) if has_both_classes(y_va) else np.nan

    print(f"\n{direction_name} - FINAL OOS (train={len(train_df)}/valid={len(valid_df)}) ACC: {acc:.4f}")
    print(f"{direction_name} - FINAL OOS AUC:  {auc:.4f}")
    print(f"\n{direction_name} - FINAL OOS classification_report:")
    print(classification_report(y_va, pred_va))

    # XGB 内置重要性（参考用：不如 OOS permutation 稳健）
    gini_series = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # ==========================================================
    # 6) 基于聚合 OOS importance 做第一轮筛选
    # ==========================================================
    thr_abs = -np.inf if (imp_min is None) else imp_min
    thr_quantile = -np.inf if (imp_quantile is None) else imp_mean.quantile(imp_quantile)
    thr = max(thr_abs, thr_quantile)

    print(
        f"\n{direction_name} - OOS perm-imp threshold: max(imp_min={thr_abs:.6f}, "
        f"quantile={thr_quantile:.6f}) = {thr:.6f}"
    )

    # 通过阈值的候选特征（按 imp_mean 降序）
    imp_candidates = imp_mean[imp_mean >= thr].sort_values(ascending=False).index.tolist()
    print(f"{direction_name} - features passing OOS importance threshold: {len(imp_candidates)}")

    # 单因子复核只做 top max_single_check（节约时间）
    single_check_features = (
        imp_candidates[:max_single_check] if (max_single_check is not None) else imp_candidates
    )
    print(
        f"{direction_name} - Running single-factor OOS AUC check for top {len(single_check_features)} features"
    )

    # ==========================================================
    # 7) 单因子 OOS AUC 复核（跨窗口取均值/最小值）
    # ==========================================================
    single_auc_by_window = {col: [] for col in single_check_features}

    if (auc_single_min is not None) and (len(single_check_features) > 0):
        print(f"\n{direction_name} - Single-factor OOS AUC check (threshold = {auc_single_min:.3f})")

        for col in single_check_features:
            j = col_to_j[col]  # 该特征在 feature_cols 中的列索引

            for w, (tr0, tr1, va0, va1) in enumerate(windows, 1):
                train_df_w = df.iloc[tr0:tr1]
                valid_df_w = df.iloc[va0:va1]

                # 只取单列特征，保持二维形状 (n, 1)
                X_tr_w = train_df_w[feature_cols].to_numpy(dtype=np.float32)[:, [j]]
                y_tr_w = train_df_w[target_col].to_numpy().astype(int)
                X_va_w = valid_df_w[feature_cols].to_numpy(dtype=np.float32)[:, [j]]
                y_va_w = valid_df_w[target_col].to_numpy().astype(int)

                # 类别不平衡权重
                n_pos_w = (y_tr_w == 1).sum()
                n_neg_w = (y_tr_w == 0).sum()
                spw_w = (n_neg_w / max(n_pos_w, 1)) if n_pos_w > 0 else 1.0

                # 单因子模型（简单些：树数上限 200，colsample=1.0）
                single_xgb = XGBClassifier(
                    n_estimators=min(200, n_estimators),
                    max_depth=max_depth,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=1.0,
                    reg_lambda=1.0,
                    min_child_weight=float(min_samples_leaf),
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    random_state=2025 + 31 * w + 7 * j,
                    scale_pos_weight=spw_w,
                    tree_method="hist",
                )

                try:
                    single_xgb.fit(X_tr_w, y_tr_w)
                    proba = single_xgb.predict_proba(X_va_w)
                    auc_w = roc_auc_score(y_va_w, proba[:, 1]) if has_both_classes(y_va_w) else np.nan
                except Exception:
                    # 某些窗口可能因为单类/数值问题训练失败，用 NaN 占位
                    auc_w = np.nan

                single_auc_by_window[col].append(auc_w)

        # 跨窗口聚合：均值与最小值（最小值可用于衡量稳定性）
        single_auc_mean = pd.Series({
            col: float(np.nanmean(aucs)) if len(aucs) else np.nan
            for col, aucs in single_auc_by_window.items()
        }).sort_values(ascending=False)

        single_auc_min = pd.Series({
            col: float(np.nanmin(aucs)) if np.any(~np.isnan(aucs)) else np.nan
            for col, aucs in single_auc_by_window.items()
        })

        print(f"\n{direction_name} - Single-factor OOS AUC mean (Top 20):")
        top20 = single_auc_mean.head(20).index
        print(pd.DataFrame({
            "auc_mean": single_auc_mean.loc[top20],
            "auc_min": single_auc_min.loc[top20],
        }))
    else:
        # 未启用单因子筛选：返回全 NaN
        single_auc_mean = pd.Series({col: np.nan for col in feature_cols})
        single_auc_min = pd.Series({col: np.nan for col in feature_cols})

    # ==========================================================
    # 8) 最终特征选择：在 base_candidates 中按单因子 AUC 阈值筛选，再截断 top_k
    # ==========================================================
    base_candidates = single_check_features

    if auc_single_min is not None:
        selected_features = [
            col for col in base_candidates
            if (col in single_auc_mean.index)
            and (not np.isnan(single_auc_mean[col]))
            and (single_auc_mean[col] >= auc_single_min)
        ]
    else:
        selected_features = list(base_candidates)

    if (top_k_max is not None) and (len(selected_features) > top_k_max):
        selected_features = selected_features[:top_k_max]

    print(f"\n{direction_name} - Final selected features: {len(selected_features)}")
    print(f"{direction_name} - Selected list (top 30 shown):")
    print(selected_features[:30])

    # ==========================================================
    # 9) 用 FINAL model 对全 df 打分（注意：严格回测不建议直接用它做全样本信号）
    # ==========================================================
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[target_col].to_numpy().astype(int)
    proba_all = model.predict_proba(X_all)
    prob_all = proba_all[:, 1]
    pred_all = (prob_all > 0.5).astype(int)

    # ==========================================================
    # 结果打包：便于后续画图/调参/接 score/backtest
    # ==========================================================
    result = {
        "model": model,
        "top_features": selected_features,         # 兼容你之前的字段命名
        "selected_features": selected_features,

        "importance_series": importance_series,    # 按 imp_mean 排序的 Series
        "importance_detail": pd.DataFrame({
            "imp_mean": imp_mean,
            "imp_median": imp_median,
            "pos_frac": imp_pos_frac,
        }).sort_values("imp_mean", ascending=False),

        "gini_importance": gini_series,            # 内置重要性（参考）
        "single_auc": single_auc_mean,             # 单因子 AUC 均值

        "acc": acc,                                # 最后窗口 valid 的 ACC
        "auc": auc,                                # 最后窗口 valid 的 AUC

        "prob": prob_all,                          # 全样本概率（更多用于诊断）
        "pred": pred_all,
        "true": y_all,
        "index": df.index,

        "windows": windows,                        # 窗口切分边界
        "window_metrics": window_metrics,          # 每窗口 valid 指标
        "single_auc_min": single_auc_min,          # 单因子 AUC 最差窗口
    }

    return result




