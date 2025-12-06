"""
Script para reentrenamiento automático basado en feedback
Ejecutar semanalmente via cron o scheduler
"""
import sys
import os
from pathlib import Path

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from app.feedback.feedback_service import feedback_service
from datetime import datetime

def main():
    print("🔄 Iniciando proceso de reentrenamiento automático")
    print(f"📅 Fecha: {datetime.now().isoformat()}")

    # 1. Obtener métricas
    print("\n📊 Obteniendo métricas...")
    metrics = feedback_service.get_metrics(days=7)

    avg_rating = metrics['general']['avg_rating']
    total_interactions = metrics['general']['total_interactions']

    print(f"   Total interacciones: {total_interactions}")
    print(f"   Rating promedio: {avg_rating}/5.0")

    # 2. Decidir si reentrenar
    THRESHOLD_RATING = 3.5
    THRESHOLD_INTERACTIONS = 100

    if avg_rating < THRESHOLD_RATING and total_interactions >= THRESHOLD_INTERACTIONS:
        print(f"\n⚠️  Rating bajo ({avg_rating}) - Iniciando reentrenamiento...")

        # 3. Exportar datos
        output_file = f"retraining_data_{datetime.now().strftime('%Y%m%d')}.jsonl"
        count = feedback_service.export_for_retraining(
            output_file=output_file,
            max_rating=3
        )

        print(f"✅ Exportados {count} ejemplos a {output_file}")
        print(f"📝 Siguiente paso: Combinar con dataset original y subir a Google Colab")
        print(f"💡 Comando: cat training_data_complete.jsonl {output_file} > training_v2.jsonl")
        print(f"📖 Ver FASE_1_FINE_TUNING_ACTUALIZADO.md para reentrenamiento")

        return count
    else:
        print(f"\n✅ Sistema funcionando bien (rating: {avg_rating})")
        print("   No es necesario reentrenar")
        return 0

if __name__ == "__main__":
    exported = main()
    sys.exit(0 if exported >= 0 else 1)