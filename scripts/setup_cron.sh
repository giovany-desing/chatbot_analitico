#!/bin/bash
#
# Script para configurar cron job de exportación semanal de datos de reentrenamiento
# Ejecutar dentro del contenedor Docker
#

set -e

echo "🔧 Configurando cron job para exportación semanal..."

# Crear directorio de logs si no existe
mkdir -p /app/logs

# Agregar entrada a crontab
# Ejecutar cada domingo a las 2 AM
CRON_JOB="0 2 * * 0 cd /app && python scripts/weekly_retraining_export.py >> /app/logs/retraining_export.log 2>&1"

# Verificar si el cron job ya existe
if crontab -l 2>/dev/null | grep -q "weekly_retraining_export.py"; then
    echo "⚠️  Cron job ya existe. Eliminando entrada anterior..."
    crontab -l 2>/dev/null | grep -v "weekly_retraining_export.py" | crontab -
fi

# Agregar nuevo cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job configurado:"
echo "   Ejecución: Cada domingo a las 2:00 AM"
echo "   Script: scripts/weekly_retraining_export.py"
echo "   Logs: logs/retraining_export.log"
echo ""

# Verificar crontab
echo "📋 Crontab actual:"
crontab -l

echo ""
echo "✅ Configuración completada"

