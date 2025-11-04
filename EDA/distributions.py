# e_distributions.py
from __future__ import annotations
import os, math
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

ART_DIR = "artifacts/eda"
os.makedirs(ART_DIR, exist_ok=True)

# ---------- A. одномерные сводки числовых ----------
def summarize_numeric(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    quantiles = [0, .01, .05, .25, .5, .75, .95, .99, 1.0],
    save_csv: str = f"{ART_DIR}/univariate_numeric.csv"
) -> pd.DataFrame:
    if exclude_cols is None:
        exclude_cols = []
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]
    rows = []
    for c in num_cols:
        s = df[c]
        q = s.quantile(quantiles)
        rows.append({
            "column": c,
            "n_missing": int(s.isna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "skew": float(s.skew()),
            "kurt": float(s.kurt()),
            "q01": float(q.get(.01, np.nan)),
            "q05": float(q.get(.05, np.nan)),
            "q25": float(q.get(.25, np.nan)),
            "q50": float(q.get(.5, np.nan)),
            "q75": float(q.get(.75, np.nan)),
            "q95": float(q.get(.95, np.nan)),
            "q99": float(q.get(.99, np.nan)),
            "min": float(q.get(0, np.nan)),
            "max": float(q.get(1.0, np.nan)),
            "zero_share": float((s==0).mean()) if (s==0).any() else 0.0
        })
    rep = pd.DataFrame(rows).sort_values("skew", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Univariate report -> {save_csv}")
    return rep

# ---------- B. выбросы и клиппинг-план ----------
def outlier_report(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    save_csv: str = f"{ART_DIR}/outlier_report.csv"
) -> pd.DataFrame:
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
        extreme_low, extreme_high = q1 - 3*iqr, q3 + 3*iqr
        frac_out = float(((s < low) | (s > high)).mean())
        frac_ext = float(((s < extreme_low) | (s > extreme_high)).mean())
        rows.append(dict(
            column=c, iqr=float(iqr),
            fence_low=float(low), fence_high=float(high),
            extreme_low=float(extreme_low), extreme_high=float(extreme_high),
            frac_out=frac_out, frac_ext=frac_ext
        ))
    rep = pd.DataFrame(rows).sort_values("frac_out", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Outlier report -> {save_csv}")
    return rep

# ---------- C. предложения трансформаций ----------
def propose_transforms(
    uni_df: pd.DataFrame,
    out_df: pd.DataFrame,
    log1p_ratio_thr: float = 50.0,   # если q95/q05 очень велик — лог
    winsorize_thr: float = 0.02      # если ≥2% за IQR — предложить клип
) -> pd.DataFrame:
    """
    uni_df: результат summarize_numeric
    out_df: результат outlier_report
    Возвращает: column | suggest
    """
    q = uni_df.set_index("column")
    o = out_df.set_index("column")
    rows = []
    for c in q.index.intersection(o.index):
        q05, q95 = q.loc[c, "q05"], q.loc[c, "q95"]
        ratio = (q95 / max(q05, 1e-9)) if pd.notna(q05) and q05 > 0 else np.inf
        skew = q.loc[c, "skew"]
        frac_out = o.loc[c, "frac_out"]
        suggest = "none"
        if ratio > log1p_ratio_thr and q05 >= 0:
            suggest = "log1p"
        if frac_out >= winsorize_thr:
            suggest = "winsorize[q01,q99]"  # можно варьировать
        # если очень тяжёлая асимметрия и много нулей — rankgauss
        if q.loc[c, "zero_share"] > 0.2 and skew > 1.0:
            suggest = "rankgauss_or_quantile"
        rows.append(dict(column=c, skew=float(skew), ratio95_05=float(ratio), out_frac=float(frac_out), suggest=suggest))
    plan = pd.DataFrame(rows).sort_values(["suggest","out_frac","skew"], ascending=[True,False,False])
    plan.to_csv(f"{ART_DIR}/transform_plan.csv", index=False)
    print(f"[INFO] Transform plan -> {ART_DIR}/transform_plan.csv")
    return plan

# ---------- D. корреляции числовых и drop-кандидаты ----------
def spearman_corr_top(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    sample: int = 120_000,
    save_csv: str = f"{ART_DIR}/corr_spearman_top.csv",
    pair_threshold: float = 0.98
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    idx = np.arange(len(df))
    if len(idx) > sample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(idx, size=sample, replace=False))
    sdf = df.loc[idx, numeric_cols].copy()
    corr = sdf.corr(method="spearman")
    corr.to_csv(save_csv)
    # пары-клоны
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= pair_threshold:
                pairs.append((cols[i], cols[j], float(r)))
    clones = pd.DataFrame(pairs, columns=["col1","col2","spearman"])
    clones.to_csv(f"{ART_DIR}/corr_spearman_pairs.csv", index=False)
    print(f"[INFO] Spearman corr -> {save_csv} | clone-pairs -> corr_spearman_pairs.csv")
    return corr, clones

# ---------- E. связь с таргетом ----------
def target_association(
    df: pd.DataFrame,
    target: str,
    problem: str = "binary",
    numeric_only: bool = True,
    save_csv: str = f"{ART_DIR}/target_assoc.csv"
) -> pd.DataFrame:
    y = df[target]
    feats = [c for c in df.columns if c != target and (pd.api.types.is_numeric_dtype(df[c]) or not numeric_only)]
    rows = []
    if problem == "binary":
        # point-biserial ~ pearson с бинарным y; AUC "фичи" как ранжирующий скор
        yb = (y == sorted(y.unique())[1]).astype(int) if y.nunique()==2 and y.dtype!=int else y.astype(int)
        for c in feats:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                try:
                    auc = roc_auc_score(yb, s)
                except Exception:
                    auc = np.nan
                corr = df[[c, target]].corr().iloc[0,1] if pd.api.types.is_numeric_dtype(df[target]) else s.corr(yb)
                rows.append(dict(feature=c, auc=float(auc) if not np.isnan(auc) else np.nan, corr=float(corr) if not np.isnan(corr) else np.nan))
        out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    elif problem == "regression":
        for c in feats:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                corr_p = df[[c, target]].corr(method="pearson").iloc[0,1]
                corr_s = df[[c, target]].corr(method="spearman").iloc[0,1]
                rows.append(dict(feature=c, pearson=float(corr_p), spearman=float(corr_s)))
        out = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    else:  # multiclass
        # mutual information (грубый ориентир)
        X = df[feats].select_dtypes(include=[np.number]).fillna(0)
        yv = y.fillna(y.mode().iloc[0])
        mi = mutual_info_classif(X, yv, discrete_features=False, random_state=42)
        out = pd.DataFrame({"feature": X.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)

    out.to_csv(save_csv, index=False)
    print(f"[INFO] Target association -> {save_csv}")
    return out

# ---------- F. выбор кандидата на дроп среди клонов ----------
def drop_clone_candidates(
    clones: pd.DataFrame,
    assoc: pd.DataFrame,
    assoc_col: str = "auc"  # 'auc' для binary, 'spearman' для regression
) -> pd.DataFrame:
    score = assoc.set_index("feature")[assoc_col]
    rows = []
    for _, r in clones.iterrows():
        a, b, rho = r["col1"], r["col2"], r["spearman"]
        sa = float(score.get(a, np.nan))
        sb = float(score.get(b, np.nan))
        # дропаем тот, у кого связь с таргетом слабее (если NaN — помечаем как осторожно)
        drop = a if (sa < sb) else b
        rows.append(dict(col1=a, col2=b, spearman=float(rho), score1=sa, score2=sb, drop_candidate=drop))
    out = pd.DataFrame(rows).sort_values(["spearman"], ascending=False)
    out.to_csv(f"{ART_DIR}/drop_candidates.csv", index=False)
    print(f"[INFO] Drop candidates -> {ART_DIR}/drop_candidates.csv}")
    return out