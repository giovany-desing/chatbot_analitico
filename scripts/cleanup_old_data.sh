#!/bin/bash
# ====================================================================
# Script de limpieza automática de datos antiguos
# ====================================================================
# Este script limpia:
# - Logs de CloudWatch mayores a 7 días (ya configurado en retención)
# - Training data exportada mayor a 30 días
# - Backups de S3 mayores a 90 días (se mueven a Glacier)
# ====================================================================

set -e

echo "🧹 Iniciando limpieza de datos antiguos..."

# ============ Configuración ============
TRAINING_BUCKET="chatbot-analitico-prod-training-data"
BACKUPS_BUCKET="chatbot-analitico-prod-backups"
REGION="us-east-1"

# ============ Limpiar Training Data > 30 días ============
echo ""
echo "📊 Limpiando training data mayor a 30 días..."

# Listar y eliminar archivos mayores a 30 días
aws s3 ls s3://${TRAINING_BUCKET}/retraining/ --recursive | \
  while read -r line; do
    createDate=$(echo "$line" | awk '{print $1" "$2}')
    createDate=$(date -d "$createDate" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$createDate" +%s)
    olderThan=$(date -d "30 days ago" +%s 2>/dev/null || date -v -30d +%s)

    if [[ $createDate -lt $olderThan ]]; then
      fileName=$(echo "$line" | awk '{print $4}')
      if [[ $fileName != "" ]]; then
        echo "  Eliminando: $fileName"
        aws s3 rm s3://${TRAINING_BUCKET}/$fileName
      fi
    fi
  done

echo "✅ Training data limpiado"

# ============ Mover Backups > 90 días a Glacier ============
echo ""
echo "💾 Moviendo backups mayores a 90 días a Glacier..."

# Este paso ya está configurado con lifecycle policies en Terraform
# Solo mostramos estadísticas
TOTAL_SIZE=$(aws s3 ls s3://${BACKUPS_BUCKET}/ --recursive --summarize | grep "Total Size" | awk '{print $3}')
TOTAL_OBJECTS=$(aws s3 ls s3://${BACKUPS_BUCKET}/ --recursive --summarize | grep "Total Objects" | awk '{print $3}')

echo "  Backups totales: $TOTAL_OBJECTS archivos"
echo "  Tamaño total: $TOTAL_SIZE bytes"
echo "✅ Lifecycle policies configuradas (automático)"

# ============ Limpiar logs locales > 7 días ============
echo ""
echo "📋 Limpiando logs locales mayores a 7 días..."

if [ -d "/opt/chatbot/logs" ]; then
  find /opt/chatbot/logs -name "*.log" -type f -mtime +7 -delete
  echo "✅ Logs locales limpiados"
else
  echo "  No hay directorio de logs locales"
fi

# ============ Resumen ============
echo ""
echo "🎉 Limpieza completada exitosamente"
echo ""
echo "📊 Resumen:"
echo "  - Training data > 30 días: Eliminado"
echo "  - Backups > 90 días: Movidos a Glacier (automático)"
echo "  - Logs locales > 7 días: Eliminados"
echo "  - CloudWatch Logs: Retención de 7 días (automático)"
echo ""

