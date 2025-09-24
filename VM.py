#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convierte un JSON de lecturas (fecha_hora, litros_consumidos) en eventos unificados
y, si existe un modelo RandomForest entrenado, predice la clase de cada evento.

Entradas:
- JSON de lecturas (array de objetos o JSON Lines). Campos requeridos:
    - fecha_hora: string ISO (o epoch) de la lectura
    - litros_consumidos: float acumulado

Salidas:
- CSV con eventos: Time, End_time, Duration_s, Flow_Lmin, Litros_totales
- (si hay modelo) CSV con predicciones anexas
"""

import os, sys, json, math, argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import joblib

# --------------------------
# Configuración de proyecto
# --------------------------
BASE_DIR = Path.home() / "consumo-ml"
MODEL_DIR = BASE_DIR / "model"
OUT_DIR_DEFAULT = BASE_DIR / "data_out"
# Cambia el nombre si tu modelo tiene otro:
RF_MODEL_PATH = MODEL_DIR / "sector_eventos_RF_sklearn.joblib"  # ej.: modelos_sklearn_rf/sector_eventos_RF_sklearn.joblib

# Etiquetas ejemplo (ajústalas a tu proyecto si quieres mapear índices a texto)
LABELS_MAP = {0: "ducha", 1: "lavaplatos", 2: "inodoro", 3: "lavamanos"}

# --------------------------
# Utilidades
# --------------------------
def parse_datetime_any(x):
    """
    Convierte 'x' a pandas.Timestamp (naive en UTC). Soporta:
    - ISO strings (e.g., '2025-09-24T12:34:56Z' o '2025-09-24 12:34:56')
    - Numérico epoch (segundos)
    """
    if pd.isna(x):
        return pd.NaT
    try:
        # si viene numérico, interpretamos como epoch (seg)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return pd.to_datetime(int(x), unit="s", utc=True).tz_convert(None)
        # si viene string
        ts = pd.to_datetime(str(x), utc=True, errors="coerce")
        if ts is pd.NaT:
            return pd.NaT
        return ts.tz_convert(None)
    except Exception:
        return pd.NaT

def ensure_monotonic(df, col='fecha_hora'):
    df = df.sort_values(col).reset_index(drop=True)
    return df

def seconds_between(t1, t2):
    return int(max((t2 - t1).total_seconds(), 0))

def to_unix_seconds(ts):
    # ts es Timestamp naive (assumimos UTC-naive). Lo tratamos como UTC.
    return int(pd.Timestamp(ts, tz='UTC').timestamp())

@dataclass
class Event:
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    litros_total: float
    duration_s: int
    flow_lmin: float

def build_events_from_series(df_raw, eps=1e-9):
    """
    A partir de una serie de lecturas con:
        - fecha_hora (Timestamp)
        - litros_consumidos (acumulado)
    Detecta eventos donde Δlitros > 0. Une tramos contiguos.
    Devuelve lista de Event.
    """
    if df_raw.empty:
        return []

    df = df_raw.copy()
    df = ensure_monotonic(df, 'fecha_hora')

    # Δ de litros (negativos -> 0)
    df['delta_l'] = df['litros_consumidos'].diff().fillna(0.0)
    df.loc[df['delta_l'] < 0, 'delta_l'] = 0.0

    # Estimamos periodo de muestreo (mediana en segundos)
    dt_sec = df['fecha_hora'].diff().dt.total_seconds().dropna()
    sample_sec = float(np.median(dt_sec)) if len(dt_sec) else 3.0  # fallback 3s

    # Flag activo cuando hay consumo en ese paso
    df['activo'] = df['delta_l'] > eps

    events = []
    in_event = False
    start_idx = None
    sum_l = 0.0

    for i in range(len(df)):
        if df.loc[i, 'activo'] and not in_event:
            # inicio de evento
            in_event = True
            start_idx = i
            sum_l = float(df.loc[i, 'delta_l'])
        elif df.loc[i, 'activo'] and in_event:
            # seguimos dentro del evento
            sum_l += float(df.loc[i, 'delta_l'])
        elif (not df.loc[i, 'activo']) and in_event:
            # terminó el evento en el punto anterior (i-1)
            end_idx = i - 1
            start_ts = df.loc[start_idx, 'fecha_hora']
            # end_ts: para representar mejor la duración real, sumamos un paso al último activo
            last_ts = df.loc[end_idx, 'fecha_hora']
            end_ts = last_ts + pd.to_timedelta(sample_sec, unit='s')

            duration_s = max(int(round((end_ts - start_ts).total_seconds())), 1)
            flow_lmin = float(sum_l) / (duration_s / 60.0) if duration_s > 0 else 0.0

            events.append(Event(start_ts, end_ts, float(sum_l), duration_s, flow_lmin))

            # reset
            in_event = False
            start_idx = None
            sum_l = 0.0
        else:
            # no activo y no estamos en evento
            pass

    # Si terminamos aún dentro de un evento, lo cerramos al final + sample_sec
    if in_event and start_idx is not None:
        end_idx = len(df) - 1
        start_ts = df.loc[start_idx, 'fecha_hora']
        last_ts = df.loc[end_idx, 'fecha_hora']
        end_ts = last_ts + pd.to_timedelta(sample_sec, unit='s')
        duration_s = max(int(round((end_ts - start_ts).total_seconds())), 1)
        flow_lmin = float(sum_l) / (duration_s / 60.0) if duration_s > 0 else 0.0
        events.append(Event(start_ts, end_ts, float(sum_l), duration_s, flow_lmin))

    return events

def construct_features_for_rf(df_events):
    """
    Construye features numéricas simples para el RF a partir de eventos.
    Puedes ampliar con más (hora, día, logs, etc.) según tu entrenamiento.
    """
    df = df_events.copy()
    # Derivados de tiempo
    df['_start_dt'] = pd.to_datetime(df['Time'], unit='s', utc=True).dt.tz_convert(None)
    df['hora']       = df['_start_dt'].dt.hour
    df['minuto']     = df['_start_dt'].dt.minute
    df['segundo']    = df['_start_dt'].dt.second
    df['dia_semana'] = df['_start_dt'].dt.dayofweek

    # Derivados de caudal
    df['Flow_Ls_mean'] = df['Flow_Lmin'] / 60.0
    df['Flow_Ls_max']  = df['Flow_Ls_mean']  # placeholder si no tienes max instantáneo
    df['L_por_s']      = df['Litros_totales'] / df['Duration_s'].clip(lower=1)

    # Logs (suaves)
    df['log_Litros'] = np.log1p(df['Litros_totales'])
    df['log_Dur']    = np.log1p(df['Duration_s'])
    df['log_FlowLm'] = np.log1p(df['Flow_Lmin'])

    features = [
        "Litros_totales","Duration_s","Flow_Lmin",
        "Flow_Ls_mean","Flow_Ls_max","L_por_s",
        "hora","minuto","segundo","dia_semana",
        "log_Litros","log_Dur","log_FlowLm"
    ]
    return df, features

def load_rf_model(path: Path):
    if path.exists():
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            print(f"[WARN] No se pudo cargar el modelo RF ({path}): {e}", file=sys.stderr)
    return None

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Unifica lecturas JSON en eventos y predice con RF (si hay modelo).")
    ap.add_argument("--in", dest="input_json", required=True, help="Ruta del archivo JSON de lecturas.")
    ap.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT), help="Carpeta de salida (CSV).")
    ap.add_argument("--model", default=str(RF_MODEL_PATH), help="Ruta al modelo RF (.joblib) (opcional).")
    ap.add_argument("--eps", type=float, default=1e-9, help="Umbral para considerar Δlitros > 0.")
    args = ap.parse_args()

    inp = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar JSON (array o jsonl) ---
    if not inp.exists():
        print(f"[ERROR] No existe el archivo: {inp}", file=sys.stderr)
        sys.exit(1)

    try:
        # Intento 1: JSON array
        df_raw = pd.read_json(inp, lines=False)
    except ValueError:
        # Intento 2: JSON Lines
        df_raw = pd.read_json(inp, lines=True)

    # Normalizamos columnas esperadas
    expected_cols = {"fecha_hora", "litros_consumidos"}
    missing = expected_cols - set(df_raw.columns)
    if missing:
        # intenta detectar columnas con nombres parecidos
        alt_map = {}
        for c in df_raw.columns:
            cl = str(c).strip().lower()
            if 'fecha' in cl or 'hora' in cl or 'date' in cl or 'time' in cl:
                alt_map[c] = 'fecha_hora'
            if 'litro' in cl and ('consum' in cl or 'acum' in cl or 'total' in cl):
                alt_map[c] = 'litros_consumidos'
        if alt_map:
            df_raw = df_raw.rename(columns=alt_map)

    missing = expected_cols - set(df_raw.columns)
    if missing:
        print(f"[ERROR] Faltan columnas requeridas en el JSON: {missing}", file=sys.stderr)
        sys.exit(1)

    # Parseo de tipos
    df_raw = df_raw[['fecha_hora','litros_consumidos']].copy()
    df_raw['fecha_hora'] = df_raw['fecha_hora'].apply(parse_datetime_any)
    df_raw['litros_consumidos'] = pd.to_numeric(df_raw['litros_consumidos'], errors='coerce')
    df_raw = df_raw.dropna(subset=['fecha_hora','litros_consumidos']).reset_index(drop=True)

    if df_raw.empty:
        print("[INFO] JSON sin datos válidos.", file=sys.stderr)
        sys.exit(0)

    # --- Construir eventos ---
    events = build_events_from_series(df_raw, eps=args.eps)
    if not events:
        print("[INFO] No se detectaron eventos de consumo (Δlitros <= eps en todas las muestras).")
        # Aún así podríamos generar un CSV vacío con headers
        out_csv = out_dir / "eventos_unificados.csv"
        empty = pd.DataFrame(columns=["Time","End_time","Duration_s","Flow_Lmin","Litros_totales"])
        empty.to_csv(out_csv, index=False)
        print(f"[OK] CSV vacío generado: {out_csv}")
        sys.exit(0)

    # --- DataFrame de eventos ---
    rows = []
    for ev in events:
        rows.append({
            "Time":        to_unix_seconds(ev.start_ts),
            "End_time":    to_unix_seconds(ev.end_ts),
            "Duration_s":  int(ev.duration_s),
            "Flow_Lmin":   float(ev.flow_lmin),
            "Litros_totales": float(ev.litros_total)
        })
    df_events = pd.DataFrame(rows)

    # Guardar CSV de eventos base
    out_csv = out_dir / "eventos_unificados.csv"
    df_events.to_csv(out_csv, index=False)
    print(f"[OK] Eventos unificados guardados en: {out_csv}")

    # --- Predicción con RF (si existe el modelo) ---
    rf_path = Path(args.model)
    rf_model = load_rf_model(rf_path)
    if rf_model is None:
        print(f"[WARN] No se encontró/abrió el modelo RF en {rf_path}. Se omite predicción.")
        sys.exit(0)

    # Construcción de features para RF
    df_feats, feat_cols = construct_features_for_rf(df_events)
    X = df_feats[feat_cols].astype(float).copy()

    try:
        proba = rf_model.predict_proba(X)
        pred_idx = proba.argmax(axis=1)
        # clases del modelo (pueden ser strings); mapeamos a texto de forma segura
        if hasattr(rf_model, "classes_"):
            classes_model = list(map(str, rf_model.classes_))
            pred_txt = [classes_model[i] for i in pred_idx]
        else:
            # fallback si no existe classes_
            pred_txt = [str(i) for i in pred_idx]
        prob_pred = proba.max(axis=1)
    except Exception as e:
        print(f"[WARN] Error al predecir con RF: {e}", file=sys.stderr)
        sys.exit(0)

    # Anexar predicciones
    df_pred = df_events.copy()
    df_pred["Pred_RF"]  = pred_txt
    df_pred["Prob_RF"]  = prob_pred

    # Si tus clases originales eran índices, puedes mapear con LABELS_MAP:
    # df_pred["Pred_RF_mapeada"] = [LABELS_MAP.get(int(x), str(x)) for x in pred_txt]

    out_pred_csv = out_dir / "eventos_unificados_pred_rf.csv"
    df_pred.to_csv(out_pred_csv, index=False)
    print(f"[OK] Predicciones RF guardadas en: {out_pred_csv}")

if __name__ == "__main__":
    main()
