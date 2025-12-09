#!/bin/bash
#
# Script helper para ejecutar exportación de datos de entrenamiento manualmente
# Uso: ./scripts/export_training_data.sh [--days N] [--min-confidence X] [--min-rating N]
#

set -e

# Valores por defecto
DAYS=7
MIN_CONFIDENCE=0.8
MIN_RATING=4
OUTPUT_DIR="data/retraining"

# Parsear argumentos
while [[ $# -gt 0 ]]; do
  case $1 in
    --days)
      DAYS="$2"
      shift 2
      ;;
    --min-confidence)
      MIN_CONFIDENCE="$2"
      shift 2
      ;;
    --min-rating)
      MIN_RATING="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Uso: $0 [--days N] [--min-confidence X] [--min-rating N] [--output-dir DIR]"
      exit 1
      ;;
  esac
done

echo "📊 Exportando datos de entrenamiento..."
echo "   Días: $DAYS"
echo "   Confianza mínima: $MIN_CONFIDENCE"
echo "   Rating mínimo: $MIN_RATING"
echo "   Directorio de salida: $OUTPUT_DIR"
echo ""

# Ejecutar dentro del contenedor
docker-compose exec app python scripts/auto_export_training_data.py \
  --days "$DAYS" \
  --min-confidence "$MIN_CONFIDENCE" \
  --min-rating "$MIN_RATING" \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ Exportación completada"

