#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lee ./data/lecturas.json(.jsonl), construye eventos unificados y,
si encuentra un modelo RF en ./data/modelos_sklearn_rf/, predice.

Salidas:
- ./data_out/eventos_unificados.csv
- ./data_out/eventos_unificados_pred_rf.csv (si hay modelo RF)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ---------------- Rutas fijas relativas al archivo ----------------
PROJ_DIR = Path(__file__).resolve().parent          # raíz del repo
DATA_DIR = PROJ_DIR / "data"
MODEL_DIR  = DATA_DIR / "dataset_editado" / "modelos_sklearn_rf"
OUT_DIR = PROJ_DIR / "data_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# acepta json y jsonl
INPUT_CANDIDATES = [DATA_DIR / "lecturas.json", DATA_DIR / "lecturas.jsonl"]
RF_MODEL = MODEL_DIR / "sector_eventos_RF_sklearn.joblib"

# ---------------- Utilidades ----------------
def parse_datetime_any(x):
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return pd.to_datetime(int(x), unit="s", utc=True).tz_convert(None)
        return pd.to_datetime(str(x), utc=True, errors="coerce").tz_convert(None)
    except Exception:
        return pd.NaT

def to_unix_seconds(ts):
    return int(pd.Timestamp(ts, tz="UTC").timestamp())

def build_events(df_raw, eps=1e-9):
    df = df_raw.sort_values("fecha_hora").reset_index(drop=True)
    # delta acumulado (si baja por reset, lo recortamos a 0)
    df["delta_l"] = df["litros_consumidos"].diff().fillna(0.0)
    df.loc[df["delta_l"] < 0, "delta_l"] = 0.0
    df["activo"] = df["delta_l"] > eps

    # periodo de muestreo (mediana); fallback 3s
    dt_sec = df["fecha_hora"].diff().dt.total_seconds().dropna()
    sample_sec = float(np.median(dt_sec)) if len(dt_sec) else 3.0

    events = []
    in_event, start_idx, sum_l = False, None, 0.0

    for i in range(len(df)):
        if df.loc[i, "activo"] and not in_event:
            in_event, start_idx, sum_l = True, i, float(df.loc[i, "delta_l"])
        elif df.loc[i, "activo"] and in_event:
            sum_l += float(df.loc[i, "delta_l"])
        elif (not df.loc[i, "activo"]) and in_event:
            end_idx = i - 1
            start_ts, last_ts = df.loc[start_idx, "fecha_hora"], df.loc[end_idx, "fecha_hora"]
            end_ts = last_ts + pd.to_timedelta(sample_sec, unit="s")
            dur = max(int((end_ts - start_ts).total_seconds()), 1)
            flow = sum_l / (dur / 60.0)
            events.append((start_ts, end_ts, dur, flow, sum_l))
            in_event = False

    if in_event:
        end_idx = len(df) - 1
        start_ts, last_ts = df.loc[start_idx, "fecha_hora"], df.loc[end_idx, "fecha_hora"]
        end_ts = last_ts + pd.to_timedelta(sample_sec, unit="s")
        dur = max(int((end_ts - start_ts).total_seconds()), 1)
        flow = sum_l / (dur / 60.0)
        events.append((start_ts, end_ts, dur, flow, sum_l))
    return events

def construct_features(df_ev):
    df = df_ev.copy()
    df["_start_dt"] = pd.to_datetime(df["Time"], unit="s", utc=True).dt.tz_convert(None)
    df["hora"] = df["_start_dt"].dt.hour
    df["minuto"] = df["_start_dt"].dt.minute
    df["segundo"] = df["_start_dt"].dt.second
    df["dia_semana"] = df["_start_dt"].dt.dayofweek
    df["Flow_Ls_mean"] = df["Flow_Lmin"] / 60.0
    df["Flow_Ls_max"]  = df["Flow_Ls_mean"]    # placeholder si no hay caudal instantáneo
    df["L_por_s"]      = df["Litros_totales"] / df["Duration_s"].clip(lower=1)
    df["log_Litros"]   = np.log1p(df["Litros_totales"])
    df["log_Dur"]      = np.log1p(df["Duration_s"])
    df["log_FlowLm"]   = np.log1p(df["Flow_Lmin"])
    feats = [
        "Litros_totales","Duration_s","Flow_Lmin",
        "Flow_Ls_mean","Flow_Ls_max","L_por_s",
        "hora","minuto","segundo","dia_semana",
        "log_Litros","log_Dur","log_FlowLm"
    ]
    return df, feats

# ---------------- Main ----------------
def main():
    # 1) localizar el input
    input_json = next((p for p in INPUT_CANDIDATES if p.exists()), None)
    if not input_json:
        print(f"[ERROR] No se encontró {INPUT_CANDIDATES[0].name} ni {INPUT_CANDIDATES[1].name} en {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    # 2) cargar JSON (array o jsonl)
    try:
        df_raw = pd.read_json(input_json, lines=False)
    except ValueError:
        df_raw = pd.read_json(input_json, lines=True)

    # 3) columnas mínimas
    if "fecha_hora" not in df_raw.columns or "litros_consumidos" not in df_raw.columns:
        print("[ERROR] El JSON debe tener columnas 'fecha_hora' y 'litros_consumidos'", file=sys.stderr)
        sys.exit(1)

    # 4) parseo y limpieza
    df_raw = df_raw[["fecha_hora", "litros_consumidos"]].copy()
    df_raw["fecha_hora"] = df_raw["fecha_hora"].apply(parse_datetime_any)
    df_raw["litros_consumidos"] = pd.to_numeric(df_raw["litros_consumidos"], errors="coerce")
    df_raw = df_raw.dropna().reset_index(drop=True)
    if df_raw.empty:
        print("[INFO] JSON sin datos válidos.")
        sys.exit(0)

    # 5) eventos
    events = build_events(df_raw)
    if not events:
        print("[INFO] No se detectaron eventos de consumo.")
        sys.exit(0)

    df_events = pd.DataFrame([{
        "Time": to_unix_seconds(s),
        "End_time": to_unix_seconds(e),
        "Duration_s": int(dur),
        "Flow_Lmin": float(flow),
        "Litros_totales": float(litros)
    } for (s, e, dur, flow, litros) in events])

    out_csv = OUT_DIR / "eventos_unificados.csv"
    df_events.to_csv(out_csv, index=False)
    print(f"[OK] Eventos guardados en {out_csv}")

    # 6) predicción si hay modelo
    if RF_MODEL.exists():
        try:
            model = joblib.load(RF_MODEL)
            df_feats, feat_cols = construct_features(df_events)
            X = df_feats[feat_cols].astype(float)
            proba = model.predict_proba(X)
            pred = model.classes_[proba.argmax(axis=1)]
            df_events["Pred_RF"] = pred
            df_events["Prob_RF"] = proba.max(axis=1)
            out_pred = OUT_DIR / "eventos_unificados_pred_rf.csv"
            df_events.to_csv(out_pred, index=False)
            print(f"[OK] Predicciones guardadas en {out_pred}")
        except Exception as e:
            print(f"[WARN] No se pudo predecir con RF: {e}", file=sys.stderr)
    else:
        print(f"[WARN] No se encontró el modelo RF en {RF_MODEL}. Se omiten predicciones.")

if __name__ == "__main__":
    main()
