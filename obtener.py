#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escucha MQTT (TLS) y guarda cada medición en ./data/lecturas.jsonl

Cada línea:
{"fecha_hora": "2025-09-24T14:05:00Z", "litros_consumidos": 12.34}

Requiere: pip install paho-mqtt
"""

from pathlib import Path
import sys, json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

# ===================== Config =====================
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
CLAVES_DIR = DATA_DIR / "Claves"
JSONL_PATH = DATA_DIR / "lecturas.jsonl"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MQTT_ENDPOINT = "a30btnoaw9tzxc-ats.iot.us-east-2.amazonaws.com"   # <-- tu endpoint IoT
MQTT_PORT     = 8883
MQTT_TOPIC    = "device/data"

# Si tus fechas llegan como "YYYY-mm-dd HH:MM:SS", ponlo aquí; si ya vienen ISO, deja None
DATE_FORMAT = None  # p.ej. "%Y-%m-%d %H:%M:%S"

# ===================== Utilidades =====================
def find_iot_certs():
    """Autodetecta CA, cert y key dentro de data/Claves/."""
    if not CLAVES_DIR.exists():
        print(f"[ERROR] No existe carpeta de claves: {CLAVES_DIR}", file=sys.stderr)
        sys.exit(1)
    # CA
    ca = None
    for name in ("AmazonRootCA1.pem", "AmazonRootCA3.pem"):
        p = CLAVES_DIR / name
        if p.exists():
            ca = p
            break
    # cert y key (por patrón)
    cert = next((p for p in CLAVES_DIR.glob("*-certificate.pem.crt")), None)
    if cert is None:
        cert = next((p for p in CLAVES_DIR.glob("*-certificate.pem")), None)
    key  = next((p for p in CLAVES_DIR.glob("*-private.pem.key")), None)

    if not ca or not cert or not key:
        print("[ERROR] No se pudieron localizar todos los archivos de TLS.", file=sys.stderr)
        print(" - CA     :", ca)
        print(" - CERT   :", cert)
        print(" - KEY    :", key)
        print("Contenido de la carpeta:", CLAVES_DIR)
        for f in sorted(CLAVES_DIR.glob("*")):
            print("   -", f.name)
        sys.exit(1)

    print("[TLS] Usando archivos:")
    print("  CA  :", ca)
    print("  CERT:", cert)
    print("  KEY :", key)
    return ca, cert, key

def to_iso(s):
    """Convierte a ISO con Z si DATE_FORMAT está definido; si no, deja tal cual."""
    if DATE_FORMAT:
        dt = datetime.strptime(str(s), DATE_FORMAT).replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return str(s).replace("+00:00", "Z")

def append_jsonl(records, path: Path):
    if not records:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()

# ===================== MQTT callbacks =====================
def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Conectado (rc={rc}). Suscribiendo a {MQTT_TOPIC} ...")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    print(f"[MQTT] Mensaje en {msg.topic} ({len(msg.payload)} bytes)")
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        print("[WARN] Payload no es JSON, se ignora.")
        return

    # Acepta lista de mediciones o dict con 'data'
    if isinstance(payload, list):
        measurements = payload
    elif isinstance(payload, dict) and "data" in payload:
        measurements = payload["data"]
    else:
        print("[WARN] Estructura inesperada:", type(payload))
        return

    to_save = []
    for m in measurements:
        if not isinstance(m, dict):
            continue
        if m.get("fecha_hora") is None or m.get("litros_consumidos") is None:
            continue
        try:
            iso = to_iso(m["fecha_hora"])
            litros = float(m["litros_consumidos"])
            to_save.append({"fecha_hora": iso, "litros_consumidos": litros})
        except Exception as e:
            print("[WARN] Medición inválida:", e)

    append_jsonl(to_save, JSONL_PATH)
    if to_save:
        print(f"[OK] Añadidas {len(to_save)} mediciones a {JSONL_PATH}")

# ===================== Main =====================
def main():
    ca, cert, key = find_iot_certs()

    # Evitar warning de API v1 (si tu versión lo soporta)
    try:
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.tls_set(
        ca_certs=str(ca),
        certfile=str(cert),
        keyfile=str(key),
    )

    print(f"[MQTT] Conectando a {MQTT_ENDPOINT}:{MQTT_PORT} ...")
    client.connect(MQTT_ENDPOINT, MQTT_PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()
