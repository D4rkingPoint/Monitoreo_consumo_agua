#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lee ./data/lecturas.json(.jsonl), construye eventos unificados y,
si encuentra un modelo RF en ./data/dataset_editado/modelos_sklearn_rf/, predice.

Salidas:
- ./data_out/eventos_unificados.csv
- ./data_out/eventos_unificados_pred_rf.csv (si hay modelo RF)
- ./data_out/eventos_unificados[_pred_rf].json  (payload para API)
- ./data_out/push_results.json                  (resumen del envío)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import requests  # <-- usamos requests para el POST

# ============ Config ============ #
PROJ_DIR  = Path(__file__).resolve().parent
DATA_DIR  = PROJ_DIR / "data"
MODEL_DIR = DATA_DIR / "dataset_editado" / "modelos_sklearn_rf"
OUT_DIR   = PROJ_DIR / "data_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [DATA_DIR / "lecturas.json", DATA_DIR / "lecturas.jsonl"]
RF_MODEL = MODEL_DIR / "sector_eventos_RF_sklearn.joblib"

# Envío a API
HOME_ID  = 1
PUSH_URL = "http://rules-instead.gl.at.ply.gg:3244/api/events/pushEvent"
TIMEOUT_S = 20

# ============ Utilidades ============ #
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
    df["delta_l"] = df["litros_consumidos"].diff().fillna(0.0)
    df.loc[df["delta_l"] < 0, "delta_l"] = 0.0
    df["activo"] = df["delta_l"] > eps
    dt_sec = df["fecha_hora"].diff().dt.total_seconds().dropna()
    sample_sec = float(np.median(dt_sec)) if len(dt_sec) else 3.0
    events, in_event, start_idx, sum_l = [], False, None, 0.0
    for i in range(len(df)):
        if df.loc[i,"activo"] and not in_event:
            in_event, start_idx, sum_l = True, i, float(df.loc[i,"delta_l"])
        elif df.loc[i,"activo"] and in_event:
            sum_l += float(df.loc[i,"delta_l"])
        elif (not df.loc[i,"activo"]) and in_event:
            end_idx = i-1
            start_ts, last_ts = df.loc[start_idx,"fecha_hora"], df.loc[end_idx,"fecha_hora"]
            end_ts = last_ts + pd.to_timedelta(sample_sec, unit="s")
            dur = max(int((end_ts-start_ts).total_seconds()),1)
            flow = sum_l / (dur/60.0)
            events.append((start_ts,end_ts,dur,flow,sum_l))
            in_event = False
    if in_event:
        end_idx = len(df)-1
        start_ts, last_ts = df.loc[start_idx,"fecha_hora"], df.loc[end_idx,"fecha_hora"]
        end_ts = last_ts + pd.to_timedelta(sample_sec, unit="s")
        dur = max(int((end_ts-start_ts).total_seconds()),1)
        flow = sum_l / (dur/60.0)
        events.append((start_ts,end_ts,dur,flow,sum_l))
    return events

def construct_features(df_ev):
    df = df_ev.copy()
    df["_start_dt"] = pd.to_datetime(df["Time"], unit="s", utc=True).dt.tz_convert(None)
    df["hora"] = df["_start_dt"].dt.hour
    df["minuto"] = df["_start_dt"].dt.minute
    df["segundo"] = df["_start_dt"].dt.second
    df["dia_semana"] = df["_start_dt"].dt.dayofweek
    df["Flow_Ls_mean"] = df["Flow_Lmin"] / 60.0
    df["Flow_Ls_max"]  = df["Flow_Ls_mean"]
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

def df_to_event_payloads(df, home_id=1):
    payloads = []
    for _, r in df.iterrows():
        payloads.append({
            "homeId": int(home_id),
            "time": int(r["Time"]),
            "endTime": int(r["End_time"]),
            "duration": float(r["Duration_s"]),
            "flow": float(r["Flow_Lmin"]),
            "totalLiters": float(r["Litros_totales"]),
            "predRF": None if "Pred_RF" not in df.columns or pd.isna(r.get("Pred_RF", None)) else str(r["Pred_RF"]),
            "probRF": None if "Prob_RF" not in df.columns or pd.isna(r.get("Prob_RF", None)) else float(r["Prob_RF"]),
        })
    return payloads

def push_all_events(events, url=PUSH_URL, timeout=TIMEOUT_S):
    """
    Intenta enviar todo el arreglo en un único POST.
    Si la API no acepta arrays, hace fallback a enviar 1 por 1.
    Devuelve un dict resumen y guarda detalle en data_out/push_results.json
    """
    headers = {"Content-Type": "application/json"}
    result = {"mode": None, "ok": 0, "fail": 0, "responses": []}

    # 1) intento bulk
    try:
        r = requests.post(url, json=events, headers=headers, timeout=timeout)
        result["responses"].append({"status": r.status_code, "text": r.text[:300], "bulk": True})
        if r.ok:
            result["mode"] = "bulk"
            result["ok"] = 1
            # guardar resumen
            with open(OUT_DIR / "push_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[PUSH] Bulk OK ({r.status_code}).")
            return result
        print(f"[PUSH] Bulk rechazado ({r.status_code}). Haciendo fallback item-by-item…")
    except Exception as e:
        result["responses"].append({"error": f"bulk_exception: {e.__class__.__name__}: {e}"})
        print("[PUSH] Bulk exception, fallback item-by-item…", e)

    # 2) fallback item-by-item
    result["mode"] = "per_item"
    for ev in events:
        try:
            r = requests.post(url, json=ev, headers=headers, timeout=timeout)
            ok = r.ok
            result["ok"] += 1 if ok else 0
            result["fail"] += 0 if ok else 1
            result["responses"].append({"status": r.status_code, "text": r.text[:300], "bulk": False})
            # pequeño respiro para no saturar
            time.sleep(0.05)
        except Exception as e:
            result["fail"] += 1
            result["responses"].append({"error": f"item_exception: {e.__class__.__name__}: {e}", "bulk": False})

    with open(OUT_DIR / "push_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[PUSH] Envío per_item: OK={result['ok']}  FAIL={result['fail']}")
    return result

# ============ Main ============ #
def main():
    input_json = next((p for p in INPUT_CANDIDATES if p.exists()), None)
    if not input_json:
        print(f"[ERROR] No se encontró {INPUT_CANDIDATES[0].name} ni {INPUT_CANDIDATES[1].name} en {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    try:
        df_raw = pd.read_json(input_json, lines=False)
    except ValueError:
        df_raw = pd.read_json(input_json, lines=True)

    if "fecha_hora" not in df_raw.columns or "litros_consumidos" not in df_raw.columns:
        print("[ERROR] El JSON debe tener columnas 'fecha_hora' y 'litros_consumidos'", file=sys.stderr)
        sys.exit(1)

    df_raw = df_raw[["fecha_hora", "litros_consumidos"]].copy()
    df_raw["fecha_hora"] = df_raw["fecha_hora"].apply(parse_datetime_any)
    df_raw["litros_consumidos"] = pd.to_numeric(df_raw["litros_consumidos"], errors="coerce")
    df_raw = df_raw.dropna().reset_index(drop=True)
    if df_raw.empty:
        print("[INFO] JSON sin datos válidos.")
        sys.exit(0)

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

    # Guardar CSV base
    out_csv = OUT_DIR / "eventos_unificados.csv"
    df_events.to_csv(out_csv, index=False)
    print(f"[OK] Eventos guardados en {out_csv}")

    # Predicción si hay modelo (antes de armar JSON)
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

    # Construir payload JSON (después de posibles predicciones)
    events_payload = df_to_event_payloads(df_events, home_id=HOME_ID)
    OUT_JSON = OUT_DIR / ("eventos_unificados_pred_rf.json" if "Pred_RF" in df_events.columns else "eventos_unificados.json")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(events_payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON guardado en: {OUT_JSON}")

    # Enviar TODO el JSON a la API (bulk y/o per-item)
    res = push_all_events(events_payload, url=PUSH_URL, timeout=TIMEOUT_S)
    print("[PUSH] Resumen:", {"mode": res["mode"], "ok": res["ok"], "fail": res["fail"]})

if __name__ == "__main__":
    main()
