# geo.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

ART_DIR = "artifacts/eda"
os.makedirs(ART_DIR, exist_ok=True)

# --- A. Поиск кандидатов lat/lon по именам и диапазонам ---
def detect_lat_lon(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cand = []
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s): 
            continue
        mn, mx = s.min(skipna=True), s.max(skipna=True)
        if mn >= -90 and mx <= 90:
            cand.append(("lat", c))
        if mn >= -180 and mx <= 180:
            cand.append(("lon", c))
    # по именам повышаем приоритет
    lat = next((c for t, c in cand if t=="lat" and "lat" in c.lower()), None)
    lon = next((c for t, c in cand if t=="lon" and ("lon" in c.lower() or "lng" in c.lower())), None)
    # если по именам не нашли — возьмем первые подходящие
    if lat is None:
        lat = next((c for t, c in cand if t=="lat"), None)
    if lon is None:
        lon = next((c for t, c in cand if t=="lon"), None)
    return lat, lon

# --- B. Haversine расстояние (км) ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p = np.pi/180
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# --- C. Профиль гео: границы, центр, NaN, «нулевые точки», примерные масштабы ---
def geo_profile(df: pd.DataFrame, lat_col: str, lon_col: str, save_json=f"{ART_DIR}/geo_profile.json") -> Dict[str, float]:
    lat = df[lat_col]; lon = df[lon_col]
    info = {
        "lat_min": float(lat.min(skipna=True)), "lat_max": float(lat.max(skipna=True)),
        "lon_min": float(lon.min(skipna=True)), "lon_max": float(lon.max(skipna=True)),
        "lat_missing": float(lat.isna().mean()), "lon_missing": float(lon.isna().mean()),
        "zero_zero_share": float(((lat==0) & (lon==0)).mean())
    }
    c_lat, c_lon = float(lat.mean(skipna=True)), float(lon.mean(skipna=True))
    info["centroid_lat"] = c_lat; info["centroid_lon"] = c_lon
    # грубая шкала: медианное расстояние до центроида на сэмпле
    idx = np.arange(len(df))
    if len(idx) > 200_000:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(idx, size=200_000, replace=False))
    d = haversine_km(df.iloc[idx][lat_col].to_numpy(), df.iloc[idx][lon_col].to_numpy(), c_lat, c_lon)
    info["median_radius_km"] = float(np.nanmedian(d))
    pd.Series(info).to_json(save_json)
    print(f"[INFO] Geo profile -> {save_json}")
    return info

# --- D. Грид (простая сетка по градусам) и плотности ---
def add_geo_grid(df: pd.DataFrame, lat_col: str, lon_col: str, res_deg: float = 0.05, prefix="g") -> pd.DataFrame:
    """
    res_deg=0.05 ≈ 5–6 км по широте; подберите под задачу (0.01~1.1км).
    """
    out = df.copy()
    out[f"{prefix}_lat_bin"] = (out[lat_col] / res_deg).round().astype("Int64")
    out[f"{prefix}_lon_bin"] = (out[lon_col] / res_deg).round().astype("Int64")
    out[f"{prefix}_cell"] = out[f"{prefix}_lat_bin"].astype(str) + "_" + out[f"{prefix}_lon_bin"].astype(str)
    return out

def grid_density(df: pd.DataFrame, cell_col: str, save_csv=f"{ART_DIR}/geo_grid_density.csv") -> pd.DataFrame:
    cnt = df[cell_col].value_counts(dropna=True)
    rep = pd.DataFrame({cell_col: cnt.index, "count": cnt.values})
    rep["freq"] = rep["count"] / rep["count"].sum()
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Geo grid density -> {save_csv}")
    return rep

# --- E. Покрытие train/test и сдвиг по клеткам (PSI/JSD упрощённые) ---
def coverage_and_shift(train: pd.DataFrame, test: pd.DataFrame, cell_col: str,
                       save_csv=f"{ART_DIR}/geo_shift_report.csv") -> pd.DataFrame:
    tr = (train[cell_col].value_counts(dropna=False) / len(train)).rename("p_tr")
    te = (test[cell_col].value_counts(dropna=False) / len(test)).rename("p_te")
    rep = pd.concat([tr, te], axis=1).fillna(0.0).reset_index().rename(columns={"index": cell_col})
    # PSI
    def psi_row(p, q):
        p = max(p, 1e-12); q = max(q, 1e-12)
        return (p - q) * np.log(p / q)
    rep["psi"] = rep.apply(lambda r: psi_row(r["p_tr"], r["p_te"]), axis=1)
    # «новые» клетки в test
    rep["is_new_in_test"] = ((rep["p_tr"]==0.0) & (rep["p_te"]>0.0)).astype(int)
    rep.to_csv(save_csv, index=False)
    print(f"[INFO] Geo shift report -> {save_csv}")
    return rep

# --- F. Простые гео-фичи: до центроида + локальная плотность клетки ---
def add_basic_geo_features(df: pd.DataFrame, lat_col: str, lon_col: str, cell_col: str,
                           centroid_lat: float, centroid_lon: float) -> pd.DataFrame:
    out = df.copy()
    out["dist_to_centroid_km"] = haversine_km(out[lat_col], out[lon_col], centroid_lat, centroid_lon)
    # локальная плотность клетки (по train-маппингу добавлять!)
    dens = out[cell_col].value_counts(normalize=True)
    out["cell_density"] = out[cell_col].map(dens).astype("float32")
    return out
