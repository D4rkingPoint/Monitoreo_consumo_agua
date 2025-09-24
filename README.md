# Monitoreo_consumo_agua

# Cómo editar y actualizar los scripts (`obtener.py` / `VM.py`)

Cuando `obtener.py` está corriendo como **servicio systemd**, el proceso se ejecuta en segundo plano todo el tiempo.  
Si editas los archivos, los cambios **no se aplican automáticamente**: hay que reiniciar el servicio para que tome la nueva versión.

---

## 🔹 Pasos para editar y aplicar cambios

### 1. Detener el servicio temporalmente

```bash
sudo systemctl stop obtener.service
```

### 2. Editar el archivo deseado
```bash
nano obtener.py  # o VM.py, según corresponda
```

# luego se puede probar manual

### 3. Reiniciar el servicio para aplicar cambios
```bash
sudo systemctl start obtener.service
```

### 4. Verificar que el servicio esté activo
```bash
sudo systemctl status obtener.service
```