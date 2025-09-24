#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Escucha MQTT (TLS), guarda SOLO el lote NUEVO en ./data/lecturas.jsonl (sobrescribe)
y dispara VM.py cuando llegan mediciones. Evita solapamientos y usa debounce.
"""

from pathlib import Path
import sys, json, time, subprocess, threading, os
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

# ===================== Config =====================
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
CLAVES_DIR = DATA_DIR / "Claves"
JSONL_PATH = DATA_DIR / "lecturas.jsonl"   # <-- JSON Lines (1 medición por línea)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# VM.py a ejecutar (misma carpeta del repo)
VM_SCRIPT = BASE_DIR / "VM.py"

# MQTT
MQTT_ENDPOINT = "a30btnoaw9tzxc-ats.iot.us-east-2.amazonaws.com"   # <-- tu endpoint IoT
MQTT_PORT     = 8883
MQTT_TOPIC    = "device/data"

# Si tus fechas llegan como "YYYY-mm-dd HH:MM:SS", coloca el formato; si ya son ISO, deja None
DATE_FORMAT = None  # p.ej. "%Y-%m-%d %H:%M:%S"

# Debounce para agrupar mensajes antes de lanzar VM.py
DEBOUNCE_SECONDS = 5

# ===================== Autodetección de certs =====================
def find_iot_certs():
    if not CLAVES_DIR.exists():
        print(f"[ERROR] No existe carpeta de claves: {CLAVES_DIR}", file=sys.stderr)
        sys.exit(1)

    ca = None
    for name in ("AmazonRootCA1.pem", "AmazonRootCA3.pem"):
        p = CLAVES_DIR / name
        if p.exists():
            ca = p
            break

    cert = next((p for p in CLAVES_DIR.glob("*-certificate.pem.crt")), None) \
        or next((p for p in CLAVES_DIR.glob("*-certificate.pem")), None)
    key  = next((p for p in CLAVES_DIR.glob("*-private.pem.key")), None)

    if not ca or not cert or not key:
        print("[ERROR] No se pudieron localizar todos los archivos TLS.", file=sys.stderr)
        print(" - CA  :", ca)
        print(" - CERT:", cert)
        print(" - KEY :", key)
        print("Contenido de", CLAVES_DIR)
        for f in sorted(CLAVES_DIR.glob("*")):
            print("   -", f.name)
        sys.exit(1)

    print("[TLS] Usando:", ca.name, cert.name, key.name)
    return ca, cert, key

# ===================== Utilidades =====================
def to_iso(s):
    """Normaliza a ISO (terminado en Z) si DATE_FORMAT está definido."""
    if DATE_FORMAT:
        dt = datetime.strptime(str(s), DATE_FORMAT).replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return str(s).replace("+00:00", "Z")

def write_jsonl_atomic(records, path: Path):
    """
    SOBRESCRIBE un .jsonl con SOLO el lote recibido (no append).
    Usa write + fsync + rename para que VM.py nunca lea un archivo a medio escribir.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)  # rename atómico en el mismo directorio

# ===================== Worker que lanza VM.py =====================
_trigger_event = threading.Event()
_running_lock  = threading.Lock()  # asegura una sola VM.py a la vez

def vm_runner_loop():
    while True:
        _trigger_event.wait()            # espera señal
        time.sleep(DEBOUNCE_SECONDS)     # debounce para agrupar mensajes
        _trigger_event.clear()

        # evita solapamiento
        if not _running_lock.acquire(blocking=False):
            # ya hay una ejecución corriendo
            continue

        try:
            if not VM_SCRIPT.exists():
                print(f"[ERROR] No existe VM.py en: {VM_SCRIPT}", file=sys.stderr)
                continue
            python = sys.executable  # mismo intérprete
            print(f"[RUN] Lanzando predicción: {python} {VM_SCRIPT}")
            proc = subprocess.run([python, str(VM_SCRIPT)],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True)
            print("[RUN][salida]\n" + proc.stdout)
            if proc.returncode != 0:
                print(f"[RUN] VM.py terminó con código {proc.returncode}", file=sys.stderr)
        finally:
            _running_lock.release()

# arrancar hilo background
threading.Thread(target=vm_runner_loop, daemon=True).start()

# ===================== MQTT callbacks =====================
def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Conectado (rc={rc}) → suscribiendo {MQTT_TOPIC}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    print(f"[MQTT] Mensaje en {msg.topic} ({len(msg.payload)} bytes)")
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        print("[WARN] Payload no es JSON")
        return

    # Aceptar lista o dict con 'data'
    measurements = payload if isinstance(payload, list) else payload.get("data", [])
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

    if to_save:
        # SOBRESCRIBE el archivo con SOLO el lote nuevo
        write_jsonl_atomic(to_save, JSONL_PATH)
        print(f"[OK] Guardado lote de {len(to_save)} mediciones en {JSONL_PATH}")
        _trigger_event.set()   # ← Señal: correr VM.py

# ===================== Main =====================
def main():
    ca, cert, key = find_iot_certs()
    try:
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message
    client.tls_set(ca_certs=str(ca), certfile=str(cert), keyfile=str(key))

    print(f"[MQTT] Conectando a {MQTT_ENDPOINT}:{MQTT_PORT}…")
    client.connect(MQTT_ENDPOINT, MQTT_PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()
