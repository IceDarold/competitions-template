# missing.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

ART_DIR = "artifacts/eda"
os.makedirs(ART_DIR, exist_ok=True)

def missing_report(df: pd.DataFrame, save_csv: str = f"{ART_DIR}/missing_report.csv") -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        miss = s.isna()
        rows.append(dict(
            column=c,
            dtype=str(s.dtype),
            n_missing=int(miss.sum()),
            missing_rate=float(miss.mean()),
            nunique=int(s.nunique(dropna=True))
        ))
    rep = pd.DataFrame(rows).sort_by("missing_rate", ascending=False) if hasattr(pd.DataFrame, "sort_by") \
          else pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Saved missing report -> {save_csv}")
    return rep

def suggest_imputation_plan(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    cat_unknown_token: str = "Unknown",
    missing_rate_threshold_indicator: float = 0.01
) -> pd.DataFrame:
    """
    Возвращает таблицу-план:
    column | type | strategy | indicator | notes
    """
    plan = []
    for c in df.columns:
        s = df[c]
        miss_rate = float(s.isna().mean())
        if pd.api.types.is_numeric_dtype(s):
            strategy = numeric_strategy
        elif pd.api.types.is_datetime64_any_dtype(s):
            strategy = "none"  # обычно делаем индикатор + возможно ffill в TS
        else:
            strategy = f"fill:{cat_unknown_token}"
        plan.append(dict(
            column=c,
            type=("numeric" if pd.api.types.is_numeric_dtype(s)
                  else "datetime" if pd.api.types.is_datetime64_any_dtype(s)
                  else "categorical"),
            strategy=strategy,
            indicator=bool(miss_rate >= missing_rate_threshold_indicator),
            notes=""
        ))
    plan_df = pd.DataFrame(plan)
    plan_df.to_csv(f"{ART_DIR}/impute_plan.csv", index=False)
    print(f"[INFO] Imputation plan -> {ART_DIR}/impute_plan.csv")
    return plan_df

def apply_imputation(
    df: pd.DataFrame,
    plan_df: pd.DataFrame,
    group_cols_for_ts: Optional[List[str]] = None,
    ts_col: Optional[str] = None,
    cat_unknown_token: str = "Unknown"
) -> pd.DataFrame:
    """
    Простая импутация по плану. Для datetime ничего не делаем (только индикаторы),
    для TS — предполагаем, что лаги/ffill будут строиться позже безопасно.
    """
    out = df.copy()
    # индикаторы:
    for _, row in plan_df.iterrows():
        c = row["column"]
        if row.get("indicator", False):
            out[f"{c}__isna"] = out[c].isna().astype("int8")

    for _, row in plan_df.iterrows():
        c = row["column"]; strat = row["strategy"]; typ = row["type"]
        if strat == "none":
            continue
        if typ == "categorical" and strat.startswith("fill:"):
            token = strat.split(":", 1)[1] or cat_unknown_token
            out[c] = out[c].astype(object).fillna(token)
        elif typ == "numeric":
            if strat == "median":
                out[c] = out[c].fillna(out[c].median())
            elif strat == "mean":
                out[c] = out[c].fillna(out[c].mean())
            elif strat.startswith("quantile:"):
                q = float(strat.split(":")[1])
                out[c] = out[c].fillna(out[c].quantile(q))
            else:
                # fallback
                out[c] = out[c].fillna(out[c].median())
        # datetime — пропускаем (индикатор уже добавили), обработка в TS-фичах
    return out

def comissing_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    sample: int = 100_000,
    top_k_by_missing: int = 40,
    save_csv: str = f"{ART_DIR}/comissing_matrix.csv"
) -> pd.DataFrame:
    """
    Быстрый co-missing: P(na_i & na_j).
    Берём топ-K колонок по missing_rate и сэмпл строк.
    """
    miss = df.isna().mean().sort_values(ascending=False)
    if cols is None:
        cols = miss.head(top_k_by_missing).index.tolist()
    idx = np.arange(len(df))
    if len(idx) > sample:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(idx, size=sample, replace=False))
    M = df.loc[idx, cols].isna().astype("int8")
    # вероятность совместного пропуска
    co = (M.T @ M) / len(M)
    co = pd.DataFrame(co, index=cols, columns=cols)
    co.to_csv(save_csv)
    print(f"[INFO] Saved co-missing matrix -> {save_csv}")
    return co

if __name__ == "__main__":
    # df = pd.read_parquet("data/train.parquet")
    # rep = missing_report(df)
    # plan = suggest_imputation_plan(df)
    # df2 = apply_imputation(df, plan)
    # comat = comissing_matrix(df)
    pass
