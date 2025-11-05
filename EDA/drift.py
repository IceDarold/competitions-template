# h_drift_v2.py
# Полная, устойчиво-работающая версия drift-модуля:
# - Импутация пропусков ПЕРЕД adversarial validation (через модуль C или fallback)
# - Безопасный препроцесс для числовых/категориальных
# - Adversarial AUC + агрегированные "важности" фич
# - PSI/JSD со стабильной обработкой NaN/бинов
from __future__ import annotations
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ART_DIR = "artifacts/eda"
os.makedirs(ART_DIR, exist_ok=True)

# === Импорт плановой импутации из модуля C, с fallback ===
try:
    from missing import suggest_imputation_plan, apply_imputation
    HAVE_C_MODULE = True
except Exception as e:
    print("[WARN] missing C module not found, using fallback imputation. Error:", e)
    HAVE_C_MODULE = False
    from sklearn.impute import SimpleImputer
    def _fallback_impute(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
        cat_cols = [c for c in out.columns if (out[c].dtype == "object" or str(out[c].dtype) == "category")]
        if num_cols:
            imputer_num = SimpleImputer(strategy="median")
            out[num_cols] = imputer_num.fit_transform(out[num_cols])
        if cat_cols:
            imputer_cat = SimpleImputer(strategy="most_frequent")
            out[cat_cols] = imputer_cat.fit_transform(out[cat_cols])
        # индикаторы пропуска (упрощённо, для fallback)
        indicators = {
            f"{c}__isna": df[c].isna().astype("int8")
            for c in df.columns
            if df[c].isna().any()
        }
        if indicators:
            out = pd.concat([out, pd.DataFrame(indicators, index=df.index)], axis=1)
        return out

# === Вспомогательные ===
def _drop_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return df.drop(columns=drop) if drop else df

def _detect_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if (df[c].dtype == "object" or str(df[c].dtype) == "category")]
    # boolean -> treat as numeric (логрег сработает корректно)
    return num_cols, cat_cols

def _onehot(min_frequency: Optional[int] = 20) -> OneHotEncoder:
    # Совместимость со старыми версиями sklearn: у некоторых нет min_frequency
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")

def _get_feature_names(ct: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    # sklearn >= 1.0: у ColumnTransformer есть get_feature_names_out
    try:
        names = ct.get_feature_names_out()
        return names.tolist()
    except Exception:
        # fallback: приблизительный список
        ohe = None
        for name, trans, cols in ct.transformers_:
            if name == "cat":
                ohe = trans
        ohe_feat = []
        if hasattr(ohe, "get_feature_names_out"):
            ohe_feat = ohe.get_feature_names_out(cat_cols).tolist()
        else:
            # грубо: расширенные имена не знаем — вернём базовые
            ohe_feat = [f"{c}_<ohe>" for c in cat_cols]
        return num_cols + ohe_feat

def _expand_to_base(name: str, cat_cols: List[str]) -> str:
    # Имена OHE обычно "col_value" — берём часть до первого "_"
    # (если в названии колонки есть "_", это не идеально, но достаточно для отчёта)
    if "_" in name:
        base = name.split("_", 1)[0]
        if base in cat_cols:
            return base
    return name

# === Единая подготовка: импутация (из модуля C или fallback) и отбор колонок ===
def _impute_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    use_module_c: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    drop_cols = set(drop_cols or [])
    # Удалим datetime-колонки из фичей (их можно закодировать отдельно)
    tr = _drop_datetime_cols(train.drop(columns=[c for c in train.columns if c in drop_cols]))
    te = _drop_datetime_cols(test.drop(columns=[c for c in test.columns if c in drop_cols]))

    common = [c for c in tr.columns if c in te.columns]
    tr, te = tr[common].copy(), te[common].copy()

    if use_module_c and HAVE_C_MODULE:
        # План импутации строим по TRAIN, применяем к обеим частям (fold-safe дух соблюдён)
        plan = suggest_imputation_plan(train[common])
        tr_imp = apply_imputation(tr, plan)
        te_imp = apply_imputation(te, plan)
    else:
        tr_imp = _fallback_impute(tr)
        te_imp = _fallback_impute(te)

    # Приведём boolean к числам — логрег устойчивее
    for df_ in (tr_imp, te_imp):
        for c in df_.columns:
            if pd.api.types.is_bool_dtype(df_[c]):
                df_[c] = df_[c].astype("int8")

    return tr_imp, te_imp

# === Построение adversarial датасета ===
def make_adv_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    use_module_c: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    X_tr, X_te = _impute_train_test(train, test, drop_cols=drop_cols, use_module_c=use_module_c)
    X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    y = np.array([0]*len(X_tr) + [1]*len(X_te))
    num_cols, cat_cols = _detect_types(X)
    return X, pd.Series(y), num_cols, cat_cols

# === Adversarial validation с имутацией ===
def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    save_csv: str = f"{ART_DIR}/adversarial_importance.csv",
    use_module_c: bool = True,
) -> Dict[str, float]:
    """
    Классификатор: 0=train, 1=test. AUC>0.60 — заметный дрифт.
    Импутация пропусков обязательна (через модуль C или встроенный fallback).
    """
    X, y, num_cols, cat_cols = make_adv_dataset(train, test, drop_cols=drop_cols, use_module_c=use_module_c)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", _onehot(min_frequency=20), cat_cols),
        ],
        remainder="drop",
        n_jobs=None
    )
    clf = LogisticRegression(max_iter=300, solver="saga")  # saga дружит со sparse
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    try:
        pipe.fit(Xtr, ytr)
    except ValueError as exc:
        if "Input X contains NaN" in str(exc):
            details = {
                "train_split": [c for c in Xtr.columns if Xtr[c].isna().any()],
                "valid_split": [c for c in Xva.columns if Xva[c].isna().any()],
                "full_dataset": [c for c in X.columns if X[c].isna().any()],
            }
            culprit_info = "\n".join(
                f"{label}: {cols}" for label, cols in details.items() if cols
            ) or "Columns not detected"
            raise ValueError(
                "adversarial_validation: leftover NaNs detected after preprocessing.\n"
                f"{culprit_info}"
            ) from exc
        raise
    p = pipe.predict_proba(Xva)[:, 1]
    auc = float(roc_auc_score(yva, p))

    # Важности: агрегируем |coef| по базовым колонкам (для OHE суммируем дамми)
    feat_names = _get_feature_names(pipe.named_steps["pre"], num_cols, cat_cols)
    coefs = pipe.named_steps["clf"].coef_.ravel()
    imp = pd.DataFrame({"feature_expanded": feat_names, "abs_coef": np.abs(coefs)})
    imp["base"] = [ _expand_to_base(nm, cat_cols) for nm in imp["feature_expanded"] ]
    agg = imp.groupby("base", as_index=False)["abs_coef"].sum().sort_values("abs_coef", ascending=False)
    agg.to_csv(save_csv, index=False)
    print(f"[INFO] Adversarial AUC={auc:.4f} | saved importances -> {save_csv}")
    return {"adversarial_auc": auc, "n_features": int(len(feat_names))}

# === PSI/JSD с безопасной обработкой NaN ===
def _psi(p: float, q: float) -> float:
    p = max(p, 1e-12); q = max(q, 1e-12)
    return (p - q) * np.log(p / q)

def _safe_numeric_bins(x: np.ndarray, y: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        # квантили по train, устраняем дубликаты бинов
        edges = np.quantile(x, np.linspace(0,1,bins+1))
        edges[0], edges[-1] = -np.inf, np.inf
        # если квантили совпали → fallback на равномерные бины
        if np.unique(edges).size < edges.size:
            mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
            if mn == mx:  # константа
                edges = np.array([-np.inf, np.inf])
            else:
                edges = np.linspace(mn, mx, bins+1)
                edges[0], edges[-1] = -np.inf, np.inf
        return edges
    else:
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        if mn == mx:
            return np.array([-np.inf, np.inf])
        edges = np.linspace(mn, mx, bins+1)
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

def psi_numeric(train: pd.Series, test: pd.Series, bins: int = 10, strategy: str = "quantile") -> float:
    x = train.dropna().to_numpy()
    y = test.dropna().to_numpy()
    if len(x) == 0 or len(y) == 0:
        return 0.0
    edges = _safe_numeric_bins(x, y, bins, strategy)
    p = np.histogram(x, bins=edges)[0] / len(x)
    q = np.histogram(y, bins=edges)[0] / len(y)
    return float(np.sum([_psi(pi, qi) for pi, qi in zip(p, q)]))

def psi_categorical(train: pd.Series, test: pd.Series) -> float:
    vc_tr = (train.fillna("<NA>").astype("object").value_counts(dropna=False) / len(train))
    vc_te = (test.fillna("<NA>").astype("object").value_counts(dropna=False) / len(test))
    keys = set(vc_tr.index).union(set(vc_te.index))
    return float(sum(_psi(float(vc_tr.get(k,0.0)), float(vc_te.get(k,0.0))) for k in keys))

def psi_report(
    train: pd.DataFrame, test: pd.DataFrame, cols: List[str],
    bins: int = 10, strategy: str = "quantile",
    save_csv: str = f"{ART_DIR}/psi_report.csv",
    use_module_c: bool = True
) -> pd.DataFrame:
    # Для PSI не обязательно имутить, но для устойчивости можно применить ту же схему
    tr, te = _impute_train_test(train[cols], test[cols], drop_cols=None, use_module_c=use_module_c)
    rows = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(tr[c]) and pd.api.types.is_numeric_dtype(te[c]):
            val = psi_numeric(tr[c], te[c], bins=bins, strategy=strategy)
        else:
            val = psi_categorical(tr[c], te[c])
        rows.append({"feature": c, "psi": float(val)})
    rep = pd.DataFrame(rows).sort_values("psi", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] PSI report -> {save_csv}")
    return rep

def jsd_numeric(train: pd.Series, test: pd.Series, bins: int = 30) -> float:
    x = train.dropna().to_numpy(); y = test.dropna().to_numpy()
    if len(x)==0 or len(y)==0: return 0.0
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    if mn == mx:
        return 0.0
    edges = np.linspace(mn, mx, bins+1)
    p = np.histogram(x, bins=edges, density=True)[0] + 1e-12
    q = np.histogram(y, bins=edges, density=True)[0] + 1e-12
    m = 0.5*(p+q)
    kl = lambda a,b: np.sum(a*np.log(a/b))
    js = 0.5*kl(p,m) + 0.5*kl(q,m)
    return float(np.sqrt(js))

def jsd_categorical(train: pd.Series, test: pd.Series) -> float:
    vc_tr = (train.fillna("<NA>").astype("object").value_counts(dropna=False) + 1e-12)
    vc_te = (test.fillna("<NA>").astype("object").value_counts(dropna=False) + 1e-12)
    keys = set(vc_tr.index).union(set(vc_te.index))
    p = np.array([float(vc_tr.get(k,1e-12)) for k in keys]); p = p/p.sum()
    q = np.array([float(vc_te.get(k,1e-12)) for k in keys]); q = q/q.sum()
    m = 0.5*(p+q)
    js = 0.5*np.sum(p*np.log(p/m)) + 0.5*np.sum(q*np.log(q/m))
    return float(np.sqrt(js))

def jsd_report(
    train: pd.DataFrame, test: pd.DataFrame, cols: List[str],
    bins: int = 30,
    save_csv: str = f"{ART_DIR}/jsd_report.csv",
    use_module_c: bool = True
) -> pd.DataFrame:
    tr, te = _impute_train_test(train[cols], test[cols], drop_cols=None, use_module_c=use_module_c)
    rows = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(tr[c]) and pd.api.types.is_numeric_dtype(te[c]):
            val = jsd_numeric(tr[c], te[c], bins=bins)
        else:
            val = jsd_categorical(tr[c], te[c])
        rows.append({"feature": c, "jsd": float(val)})
    rep = pd.DataFrame(rows).sort_values("jsd", ascending=False)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] JSD report -> {save_csv}")
    return rep

def drift_summary(
    train: pd.DataFrame, test: pd.DataFrame,
    drop_cols: Optional[List[str]] = None,
    max_cols: int = 200,
    save_csv: str = f"{ART_DIR}/drift_summary.csv",
    use_module_c: bool = True
) -> pd.DataFrame:
    common = [c for c in train.columns if c in test.columns and c not in (drop_cols or [])]
    cols = common[:max_cols]
    psi = psi_report(train, test, cols, use_module_c=use_module_c)
    jsd = jsd_report(train, test, cols, use_module_c=use_module_c)
    rep = psi.merge(jsd, on="feature", how="outer")
    rep["psi_flag"] = pd.cut(rep["psi"].fillna(0.0), bins=[-1,0.1,0.25,1e9], labels=["low","medium","high"])
    rep["jsd_flag"] = pd.cut(rep["jsd"].fillna(0.0), bins=[-1,0.05,0.15,1e9], labels=["low","medium","high"])
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Drift summary -> {save_csv}")
    return rep

# === Пример использования ===
if __name__ == "__main__":
    # train = pd.read_parquet("data/train.parquet")
    # test  = pd.read_parquet("data/test.parquet")
    # adv = adversarial_validation(train, test, drop_cols=["target"] if "target" in train.columns else None)
    # common = [c for c in train.columns if c in test.columns and c != "target"]
    # psi = psi_report(train, test, common[:200])
    # jsd = jsd_report(train, test, common[:200])
    # summary = drift_summary(train, test, drop_cols=["target"] if "target" in train.columns else None)
    pass
