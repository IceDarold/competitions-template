# h_drift.py
from __future__ import annotations
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ART_DIR = "artifacts/eda"
os.makedirs(ART_DIR, exist_ok=True)

# --- A. Собираем adversarial датасет: label=1 (test), 0 (train) ---
def make_adv_dataset(train: pd.DataFrame, test: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    drop_cols = set(drop_cols or [])
    common = [c for c in train.columns if c in test.columns and c not in drop_cols]
    X = pd.concat([train[common].copy(), test[common].copy()], axis=0, ignore_index=True)
    y = np.array([0]*len(train) + [1]*len(test))
    num_cols = [c for c in common if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in common if (X[c].dtype=="object" or str(X[c].dtype)=="category")]
    return X, pd.Series(y), num_cols, cat_cols

# --- B. Adversarial validation (логрег + OHE) ---
def adversarial_validation(
    train: pd.DataFrame, test: pd.DataFrame, drop_cols: Optional[List[str]] = None,
    test_size: float = 0.25, random_state: int = 42, save_csv=f"{ART_DIR}/adversarial_importance.csv"
) -> Dict[str, float]:
    X, y, num_cols, cat_cols = make_adv_dataset(train, test, drop_cols)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # препроцесс: числовые -> passthrough, категориальные -> OHE(min_freq)
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", ohe, cat_cols)
    ], remainder="drop", n_jobs=None)

    clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xva)[:,1]
    auc = float(roc_auc_score(yva, p))

    # простая "важность" признаков для логрега через абсолютные коэф-ты
    # (для OHE восстановим важность по группам: сумма |coef| по дамми-столбцам)
    coefs = pipe.named_steps["clf"].coef_.ravel()
    ohe_feat_names = []
    if cat_cols:
        ohe_feat_names = pipe.named_steps["pre"].transformers_[1][1].get_feature_names_out(cat_cols).tolist()
    num_feat_names = num_cols
    full_names = num_feat_names + ohe_feat_names
    imp = pd.DataFrame({"feature_expanded": full_names, "abs_coef": np.abs(coefs)})
    # агрегируем по исходным фичам (для категорий — суммируем дамми)
    def base_name(s: str) -> str:
        # OHE имена вида "col__value"; вернём базовую колонку
        return s.split("_", 1)[0] if s and ("_" in s) and (s.split("_",1)[0] in cat_cols) else s.split("_", 1)[0]
    imp["base"] = [bn if (bn in num_cols or bn in cat_cols) else s.split("[",1)[0] for s, bn in zip(imp["feature_expanded"], map(base_name, imp["feature_expanded"]))]
    agg = imp.groupby("base", as_index=False)["abs_coef"].sum().sort_values("abs_coef", ascending=False)
    agg.to_csv(save_csv, index=False)
    print(f"[INFO] Adversarial AUC={auc:.4f} | saved importances -> {save_csv}")
    return {"adversarial_auc": auc, "n_features": len(full_names)}

# --- C. PSI (Population Stability Index) для числовых/категориальных ---
def _psi(p: float, q: float) -> float:
    p = max(p, 1e-12); q = max(q, 1e-12)
    return (p - q) * np.log(p / q)

def psi_numeric(train: pd.Series, test: pd.Series, bins: int = 10, strategy: str = "quantile") -> float:
    x = train.dropna().to_numpy()
    y = test.dropna().to_numpy()
    if len(x) == 0 or len(y) == 0: 
        return 0.0
    if strategy == "quantile":
        edges = np.quantile(x, np.linspace(0,1,bins+1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        edges = np.linspace(mn, mx, bins+1); edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(x, bins=edges)[0] / len(x)
    q = np.histogram(y, bins=edges)[0] / len(y)
    return float(np.sum([_psi(pi, qi) for pi, qi in zip(p, q)]))

def psi_categorical(train: pd.Series, test: pd.Series) -> float:
    vc_tr = (train.value_counts(dropna=False) / len(train))
    vc_te = (test.value_counts(dropna=False) / len(test))
    keys = set(vc_tr.index).union(set(vc_te.index))
    return float(sum(_psi(float(vc_tr.get(k,0.0)), float(vc_te.get(k,0.0))) for k in keys))

def psi_report(train: pd.DataFrame, test: pd.DataFrame, cols: List[str], bins: int = 10, strategy: str = "quantile",
               save_csv=f"{ART_DIR}/psi_report.csv") -> pd.DataFrame:
    rows = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c]):
            val = psi_numeric(train[c], test[c], bins=bins, strategy=strategy)
        else:
            val = psi_categorical(train[c].astype("object"), test[c].astype("object"))
        rows.append({"feature": c, "psi": float(val)})
    rep = pd.DataFrame(rows).sort_values("psi", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] PSI report -> {save_csv}")
    return rep

# --- D. JSD (Дженсен–Шеннон) для числовых и категориальных (быстрый) ---
def jsd_numeric(train: pd.Series, test: pd.Series, bins: int = 30) -> float:
    x = train.dropna().to_numpy(); y = test.dropna().to_numpy()
    if len(x)==0 or len(y)==0: return 0.0
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    edges = np.linspace(mn, mx, bins+1)
    p = np.histogram(x, bins=edges, density=True)[0] + 1e-12
    q = np.histogram(y, bins=edges, density=True)[0] + 1e-12
    m = 0.5*(p+q)
    kl = lambda a,b: np.sum(a*np.log(a/b))
    js = 0.5*kl(p,m) + 0.5*kl(q,m)
    return float(np.sqrt(js))  # JS расстояние

def jsd_categorical(train: pd.Series, test: pd.Series) -> float:
    vc_tr = (train.value_counts(dropna=False) / len(train)) + 1e-12
    vc_te = (test.value_counts(dropna=False) / len(test)) + 1e-12
    keys = set(vc_tr.index).union(set(vc_te.index))
    p = np.array([float(vc_tr.get(k,1e-12)) for k in keys])
    q = np.array([float(vc_te.get(k,1e-12)) for k in keys])
    p, q = p/p.sum(), q/q.sum()
    m = 0.5*(p+q)
    js = 0.5*np.sum(p*np.log(p/m)) + 0.5*np.sum(q*np.log(q/m))
    return float(np.sqrt(js))

def jsd_report(train: pd.DataFrame, test: pd.DataFrame, cols: List[str], bins: int = 30,
               save_csv=f"{ART_DIR}/jsd_report.csv") -> pd.DataFrame:
    rows = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c]):
            val = jsd_numeric(train[c], test[c], bins=bins)
        else:
            val = jsd_categorical(train[c].astype("object"), test[c].astype("object"))
        rows.append({"feature": c, "jsd": float(val)})
    rep = pd.DataFrame(rows).sort_values("jsd", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] JSD report -> {save_csv}")
    return rep

# --- E. Сводный отчёт и подсказки ---
def drift_summary(train: pd.DataFrame, test: pd.DataFrame, drop_cols: Optional[List[str]] = None,
                  max_cols: int = 200, save_csv=f"{ART_DIR}/drift_summary.csv") -> pd.DataFrame:
    X, _, _, _ = make_adv_dataset(train, test, drop_cols)
    # ограничим число колонок для отчётов
    cols = [c for c in X.columns if c not in (drop_cols or [])][:max_cols]
    psi = psi_report(train, test, cols)
    jsd = jsd_report(train, test, cols)
    rep = psi.merge(jsd, on="feature", how="outer")
    # эвристики по степени риска
    rep["psi_flag"] = pd.cut(rep["psi"].fillna(0.0), bins=[-1,0.1,0.25,1e9], labels=["low","medium","high"])
    rep["jsd_flag"] = pd.cut(rep["jsd"].fillna(0.0), bins=[-1,0.05,0.15,1e9], labels=["low","medium","high"])
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Drift summary -> {save_csv}")
    return rep
