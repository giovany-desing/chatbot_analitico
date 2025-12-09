# Configuración de Exportación Automática de Datos de Reentrenamiento

## Variables de Entorno

Agregar al archivo `.env`:

```bash
# Exportación automática de datos de reentrenamiento
RETRAINING_EXPORT_ENABLED=true
RETRAINING_EXPORT_INTERVAL_DAYS=7
RETRAINING_EXPORT_MIN_SAMPLES=50
RETRAINING_CLEANUP_DAYS=30
RETRAINING_KEEP_LAST_N=5
```

### Descripción de Variables

- **RETRAINING_EXPORT_ENABLED**: Habilitar/deshabilitar exportación automática (default: `true`)
- **RETRAINING_EXPORT_INTERVAL_DAYS**: Días entre exportaciones (default: `7`)
- **RETRAINING_EXPORT_MIN_SAMPLES**: Mínimo de ejemplos requeridos para exportar (default: `50`)
- **RETRAINING_CLEANUP_DAYS**: Días antes de eliminar archivos antiguos (default: `30`)
- **RETRAINING_KEEP_LAST_N**: Número de archivos más recientes a mantener (default: `5`)

## Uso del Script

### Modo Manual (forzar exportación ahora)

```bash
python scripts/weekly_retraining_export.py --force
```

### Modo Automático (solo si han pasado 7 días)

```bash
python scripts/weekly_retraining_export.py
```

### Dry-run (simular sin exportar)

```bash
python scripts/weekly_retraining_export.py --dry-run
```

### Especificar días personalizados

```bash
python scripts/weekly_retraining_export.py --days 14
```

## Configuración de Cron (Docker)

### Opción 1: Usar el script de setup

```bash
# Dentro del contenedor
docker-compose exec app bash scripts/setup_cron.sh
```

### Opción 2: Configurar manualmente

```bash
# Dentro del contenedor
docker-compose exec app bash

# Agregar cron job
echo "0 2 * * 0 cd /app && python scripts/weekly_retraining_export.py >> /app/logs/retraining_export.log 2>&1" | crontab -

# Verificar
crontab -l
```

### Opción 3: Usar servicio Docker (recomendado)

El servicio `cron-retraining` en `docker-compose.yml` ya está configurado para ejecutar el script semanalmente.

## Logs

Los logs se guardan en:
- `logs/retraining_export.log`

Ver logs en tiempo real:
```bash
tail -f logs/retraining_export.log
```

## Validaciones Pre-Exportación

El script valida automáticamente:

1. ✅ Datos recientes en PostgreSQL (mínimo `RETRAINING_EXPORT_MIN_SAMPLES`)
2. ✅ Espacio en disco disponible (>100MB)
3. ✅ Success rate de métricas (>70%)

Si alguna validación falla, la exportación se aborta y se registra un warning.

## Cleanup Automático

El script elimina automáticamente:

- Archivos más antiguos que `RETRAINING_CLEANUP_DAYS` días
- Mantiene solo los `RETRAINING_KEEP_LAST_N` archivos más recientes

## Métricas

Cada exportación registra métricas en `performance_metrics`:

- Component: `retraining_export`
- Success: `true/false`
- Latency: tiempo de exportación en ms
- Metadata: estadísticas de la exportación

## Troubleshooting

### Error: "Solo hay X ejemplos recientes"

- Verificar que hay actividad reciente en el sistema
- Reducir `RETRAINING_EXPORT_MIN_SAMPLES` si es necesario
- Aumentar `RETRAINING_EXPORT_INTERVAL_DAYS` para acumular más datos

### Error: "Espacio en disco insuficiente"

- Limpiar archivos antiguos manualmente
- Aumentar `RETRAINING_CLEANUP_DAYS` para limpiar más agresivamente
- Verificar espacio en el volumen de Docker

### Error: "Success rate muy bajo"

- Revisar logs de errores del sistema
- Verificar que los componentes están funcionando correctamente
- Considerar ajustar umbrales de validación

