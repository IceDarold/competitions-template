# d_categoricals.py
from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

ART_DIR = "artifacts/eda"
os.makedirs(ART_DIR, exist_ok=True)

# ---------- отчёт по кардинальности ----------
def cardinality_report(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    save_csv: str = f"{ART_DIR}/cardinality_report.csv"
) -> pd.DataFrame:
    if cat_cols is None:
        cat_cols = [c for c in df.columns
                    if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    rows = []
    for c in cat_cols:
        nunq = int(df[c].nunique(dropna=True))
        top = df[c].value_counts(dropna=True).head(5).to_dict()
        rows.append(dict(column=c, nunique=nunq, top5=top))
    rep = pd.DataFrame(rows).sort_values("nunique", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Cardinality report -> {save_csv}")
    return rep

# ---------- редкие категории ----------
def rare_binning(
    s: pd.Series, min_count: int = 20, other_label: str = "OTHER"
) -> pd.Series:
    vc = s.value_counts(dropna=False)
    rare_vals = vc[vc < min_count].index
    out = s.copy()
    out = out.where(~out.isin(rare_vals), other_label)
    return out

# ---------- частотные кодировки (frequency) ----------
class FreqEncoder:
    def __init__(self):
        self.maps: Dict[str, Dict[object, float]] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        for c in cat_cols:
            freq = (df[c].value_counts(dropna=True) / df.shape[0]).to_dict()
            self.maps[c] = freq
        return self

    def transform(self, df: pd.DataFrame, cat_cols: List[str], other_value: float = 0.0) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            m = self.maps.get(c, {})
            out[f"{c}__freq"] = out[c].map(m).fillna(other_value).astype("float32")
        return out

# ---------- fold-safe target encoding с сглаживанием ----------
class KFoldTargetEncoder:
    def __init__(self, n_splits: int = 5, smoothing: float = 20.0, random_state: int = 42, stratify: Optional[pd.Series] = None):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.global_mean_: float = np.nan
        self.maps_: Dict[str, Dict[object, float]] = {}
        self.stratify = stratify

    @staticmethod
    def _mean_smooth(cnt: int, mean_cat: float, global_mean: float, m: float) -> float:
        # классическая m-сглаженная оценка
        return (cnt * mean_cat + m * global_mean) / (cnt + m)

    def fit_transform(
        self, df: pd.DataFrame, y: pd.Series, cat_cols: List[str]
    ) -> pd.DataFrame:
        out = df.copy()
        self.global_mean_ = float(y.mean())
        # сплиты
        if self.stratify is not None and len(np.unique(self.stratify)) > 1:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = skf.split(df, self.stratify)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = kf.split(df)

        oof_maps_per_col: Dict[str, Dict[int, Dict[object, float]]] = {c: {} for c in cat_cols}
        oof_encoded = {c: np.zeros(len(df), dtype="float32") for c in cat_cols}

        for fold, (tr, va) in enumerate(splits):
            y_tr = y.iloc[tr]
            for c in cat_cols:
                gp = df.iloc[tr].groupby(c)[y.name].agg(["count", "mean"]).rename(columns={"mean": "m", "count": "n"})
                gp["enc"] = self._mean_smooth(gp["n"], gp["m"], self.global_mean_, self.smoothing)
                mapping = gp["enc"].to_dict()
                oof_maps_per_col[c][fold] = mapping
                oof_encoded[c][va] = df.iloc[va][c].map(mapping).fillna(self.global_mean_).astype("float32").values

        for c in cat_cols:
            out[f"{c}__te"] = oof_encoded[c]

        # финальные карты на всём df (для test/holdout)
        for c in cat_cols:
            gp = df.groupby(c)[y.name].agg(["count", "mean"]).rename(columns={"mean": "m", "count": "n"})
            gp["enc"] = self._mean_smooth(gp["n"], gp["m"], self.global_mean_, self.smoothing)
            self.maps_[c] = gp["enc"].to_dict()
        return out

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            m = self.maps_.get(c, {})
            out[f"{c}__te"] = out[c].map(m).fillna(self.global_mean_).astype("float32")
        return out

# ---------- hashing trick (стабильный, быстрый) ----------
def hashing_trick(s: pd.Series, n_buckets: int = 1024, seed: int = 13) -> pd.Series:
    # стабильный простой хэш: без внешних либ
    # важно: одинаков для train/valid/test
    return (s.astype("string").fillna("<NA>").apply(lambda x: hash((x, seed)) % n_buckets)).astype("int32")

# ---------- план кодирования ----------
def encoding_plan(
    df: pd.DataFrame,
    rare_min_count: int = 20,
    low_card_threshold: int = 10,
    mid_card_threshold: int = 2000
) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    rows = []
    for c in cat_cols:
        nunq = int(df[c].nunique(dropna=True))
        if nunq <= low_card_threshold:
            strat = "one-hot / ordinal (CatBoost-friendly)"
        elif nunq <= mid_card_threshold:
            strat = "freq + KFoldTarget(smoothing)"
        else:
            strat = "hashing_trick or CatBoost (TE аккуратно)"
        rows.append(dict(column=c, nunique=nunq, strategy=strat, rare_min_count=rare_min_count))
    plan = pd.DataFrame(rows).sort_values("nunique", ascending=False)
    plan.to_csv(f"{ART_DIR}/categoricals_plan.csv", index=False)
    print(f"[INFO] Saved encoding plan -> {ART_DIR}/categoricals_plan.csv")
    return plan

if __name__ == "__main__":
    pass
