"""
Script de monitoreo semanal para exportación automática de datos de reentrenamiento.
Verifica si han pasado 7 días desde la última exportación y ejecuta exportación si es necesario.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import logging
import shutil
import argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from sqlalchemy import text
import time

from app.db.connections import get_postgres
from app.metrics.performance_tracker import track_hybrid_execution
from scripts.auto_export_training_data import export_training_data

# Configuración por defecto (puede ser sobrescrita por .env)
RETRAINING_EXPORT_ENABLED = os.getenv("RETRAINING_EXPORT_ENABLED", "true").lower() == "true"
RETRAINING_EXPORT_INTERVAL_DAYS = int(os.getenv("RETRAINING_EXPORT_INTERVAL_DAYS", "7"))
RETRAINING_EXPORT_MIN_SAMPLES = int(os.getenv("RETRAINING_EXPORT_MIN_SAMPLES", "50"))
RETRAINING_CLEANUP_DAYS = int(os.getenv("RETRAINING_CLEANUP_DAYS", "30"))
RETRAINING_KEEP_LAST_N = int(os.getenv("RETRAINING_KEEP_LAST_N", "5"))

# Configurar logging estructurado
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "retraining_export.log"

# Configurar logger
logger = logging.getLogger("retraining_export")
logger.setLevel(logging.INFO)

# Handler para archivo
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formato estructurado
formatter = logging.Formatter(
    '[RETRAINING-EXPORT] %(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ============ Gestión de Tabla de Metadata ============

def create_export_metadata_table() -> None:
    """Crea la tabla export_metadata si no existe"""
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS export_metadata (
            id SERIAL PRIMARY KEY,
            export_type VARCHAR(50) NOT NULL,
            last_export_date TIMESTAMP NOT NULL,
            records_exported INT,
            output_file TEXT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_export_type ON export_metadata(export_type);
        CREATE INDEX IF NOT EXISTS idx_last_export_date ON export_metadata(last_export_date DESC);
        """
        
        session.execute(text(create_table_sql))
        session.commit()
        session.close()
        
        logger.info("✅ Tabla export_metadata creada/verificada")
        
    except Exception as e:
        logger.error(f"❌ Error creando tabla export_metadata: {e}", exc_info=True)
        raise


def get_last_export_date(export_type: str = "retraining") -> Optional[datetime]:
    """
    Consulta export_metadata para obtener última exportación.
    
    Args:
        export_type: Tipo de exportación (default: "retraining")
    
    Returns:
        datetime de última exportación o None si nunca se ha exportado
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        sql = """
        SELECT last_export_date
        FROM export_metadata
        WHERE export_type = :export_type
        ORDER BY last_export_date DESC
        LIMIT 1
        """
        
        result = session.execute(
            text(sql),
            {'export_type': export_type}
        ).fetchone()
        
        session.close()
        
        if result and result[0]:
            return result[0]
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo última exportación: {e}", exc_info=True)
        return None


def record_export(
    export_type: str,
    records: int,
    file_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Inserta registro de exportación exitosa.
    
    Args:
        export_type: Tipo de exportación
        records: Número de registros exportados
        file_path: Ruta del archivo generado
        metadata: Metadata adicional (JSON)
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        insert_sql = """
        INSERT INTO export_metadata (
            export_type,
            last_export_date,
            records_exported,
            output_file,
            metadata
        ) VALUES (
            :export_type,
            :last_export_date,
            :records_exported,
            :output_file,
            :metadata
        )
        """
        
        session.execute(
            text(insert_sql),
            {
                'export_type': export_type,
                'last_export_date': datetime.now(),
                'records_exported': records,
                'output_file': file_path,
                'metadata': json.dumps(metadata or {})
            }
        )
        session.commit()
        session.close()
        
        logger.info(f"✅ Registro de exportación guardado: {records} registros")
        
    except Exception as e:
        logger.error(f"❌ Error guardando registro de exportación: {e}", exc_info=True)
        raise


# ============ Validaciones Pre-Exportación ============

def validate_pre_export(days: int) -> Tuple[bool, str]:
    """
    Valida condiciones antes de exportar.
    
    Args:
        days: Días hacia atrás para exportar
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # 1. Verificar que PostgreSQL tiene datos recientes
        postgres = get_postgres()
        session = postgres.get_session()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        sql = """
        SELECT COUNT(*) as count
        FROM performance_metrics
        WHERE timestamp >= :cutoff_date
            AND component IN ('viz', 'hybrid')
            AND success = TRUE
        """
        
        result = session.execute(
            text(sql),
            {'cutoff_date': cutoff_date}
        ).fetchone()
        
        session.close()
        
        recent_data_count = result[0] if result else 0
        
        if recent_data_count < RETRAINING_EXPORT_MIN_SAMPLES:
            return False, f"Solo hay {recent_data_count} ejemplos recientes (mínimo: {RETRAINING_EXPORT_MIN_SAMPLES})"
        
        # 2. Verificar espacio en disco (>100MB)
        output_dir = Path("data/retraining")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stat = shutil.disk_usage(output_dir)
        free_mb = stat.free / (1024 * 1024)
        
        if free_mb < 100:
            return False, f"Espacio en disco insuficiente: {free_mb:.1f}MB disponibles (mínimo: 100MB)"
        
        # 3. Verificar que hay datos de calidad (success rate > 0.8)
        session = postgres.get_session()
        
        sql = """
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE success = TRUE) as successful
        FROM performance_metrics
        WHERE timestamp >= :cutoff_date
            AND component IN ('viz', 'hybrid')
        """
        
        result = session.execute(
            text(sql),
            {'cutoff_date': cutoff_date}
        ).fetchone()
        
        session.close()
        
        total = result[0] if result else 0
        successful = result[1] if result else 0
        
        if total > 0:
            success_rate = successful / total
            if success_rate < 0.7:
                return False, f"Success rate muy bajo: {success_rate:.1%} (mínimo recomendado: 70%)"
        
        return True, "Validaciones pasadas"
        
    except Exception as e:
        logger.error(f"❌ Error en validaciones: {e}", exc_info=True)
        return False, f"Error en validaciones: {str(e)}"


# ============ Cleanup Automático ============

def cleanup_old_exports() -> int:
    """
    Elimina exports antiguos y mantiene solo los N más recientes.
    
    Returns:
        Número de archivos eliminados
    """
    try:
        output_dir = Path("data/retraining")
        
        if not output_dir.exists():
            return 0
        
        # Obtener todos los archivos de exportación
        export_files = list(output_dir.glob("training_data_*.jsonl"))
        
        if len(export_files) <= RETRAINING_KEEP_LAST_N:
            return 0
        
        # Ordenar por fecha de modificación (más recientes primero)
        export_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Eliminar los antiguos
        files_to_delete = export_files[RETRAINING_KEEP_LAST_N:]
        deleted_count = 0
        
        cutoff_date = datetime.now() - timedelta(days=RETRAINING_CLEANUP_DAYS)
        
        for file_path in files_to_delete:
            # Eliminar si es más antiguo que RETRAINING_CLEANUP_DAYS
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if file_mtime < cutoff_date:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️  Eliminado archivo antiguo: {file_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️  Error eliminando {file_path.name}: {e}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ Error en cleanup: {e}", exc_info=True)
        return 0


# ============ Generación de Reporte ============

def generate_export_report(output_file: str) -> Dict:
    """
    Genera reporte detallado de la exportación.
    
    Args:
        output_file: Ruta del archivo exportado
    
    Returns:
        Dict con estadísticas del reporte
    """
    try:
        report = {
            "total_samples": 0,
            "chart_type_distribution": {},
            "avg_confidence": 0.0,
            "file_size_mb": 0.0,
            "sources": {"performance_metrics": 0, "feedback": 0}
        }
        
        # Leer archivo y analizar
        chart_type_counts = {}
        confidence_values = []
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    report["total_samples"] += 1
                    
                    # Extraer chart_type
                    assistant_content = data['messages'][2]['content']
                    assistant_data = json.loads(assistant_content)
                    chart_type = assistant_data.get('chart_type', 'unknown')
                    chart_type_counts[chart_type] = chart_type_counts.get(chart_type, 0) + 1
                    
                    # Extraer confidence
                    confidence = assistant_data.get('confidence', 0.0)
                    if confidence > 0:
                        confidence_values.append(confidence)
                    
                    # Detectar fuente
                    if assistant_data.get('source') == 'user_feedback':
                        report["sources"]["feedback"] += 1
                    else:
                        report["sources"]["performance_metrics"] += 1
                        
                except Exception as e:
                    logger.warning(f"Error procesando línea del reporte: {e}")
        
        report["chart_type_distribution"] = chart_type_counts
        
        if confidence_values:
            report["avg_confidence"] = sum(confidence_values) / len(confidence_values)
        
        # Tamaño del archivo
        file_path = Path(output_file)
        if file_path.exists():
            report["file_size_mb"] = file_path.stat().st_size / (1024 * 1024)
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Error generando reporte: {e}", exc_info=True)
        return {}


# ============ Función Principal ============

def check_and_export(
    force: bool = False,
    dry_run: bool = False,
    days: Optional[int] = None
) -> bool:
    """
    Función principal: verifica y ejecuta exportación si es necesario.
    
    Args:
        force: Forzar exportación sin verificar fecha
        dry_run: Simular sin exportar realmente
        days: Días hacia atrás (default: usar configuración)
    
    Returns:
        True si exportación fue exitosa o no necesaria, False si falló
    """
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("📊 Iniciando verificación de exportación semanal")
        logger.info("=" * 60)
        
        # Verificar si está habilitado
        if not RETRAINING_EXPORT_ENABLED:
            logger.info("⚠️  Exportación automática deshabilitada (RETRAINING_EXPORT_ENABLED=false)")
            return True
        
        # Crear tabla de metadata si no existe
        create_export_metadata_table()
        
        # Determinar días a exportar
        export_days = days if days is not None else RETRAINING_EXPORT_INTERVAL_DAYS
        
        # Verificar última exportación (a menos que sea forzado)
        if not force:
            last_export = get_last_export_date()
            
            if last_export:
                days_since_last = (datetime.now() - last_export).days
                logger.info(f"📅 Última exportación: {last_export.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"📅 Días desde última exportación: {days_since_last}")
                
                if days_since_last < RETRAINING_EXPORT_INTERVAL_DAYS:
                    logger.info(f"✅ No es necesario exportar aún (faltan {RETRAINING_EXPORT_INTERVAL_DAYS - days_since_last} días)")
                    return True
            else:
                logger.info("📅 No hay exportaciones previas registradas")
        
        # Validaciones pre-exportación
        logger.info("🔍 Ejecutando validaciones pre-exportación...")
        is_valid, validation_message = validate_pre_export(export_days)
        
        if not is_valid:
            logger.warning(f"⚠️  Validaciones fallaron: {validation_message}")
            logger.warning("⚠️  Abortando exportación")
            return False
        
        logger.info(f"✅ {validation_message}")
        
        # Dry-run mode
        if dry_run:
            logger.info("🔍 DRY-RUN: Simulando exportación (no se exportará realmente)")
            logger.info(f"   Días: {export_days}")
            logger.info(f"   Mínimo de muestras: {RETRAINING_EXPORT_MIN_SAMPLES}")
            return True
        
        # Ejecutar exportación
        logger.info(f"📤 Iniciando exportación (últimos {export_days} días)...")
        
        output_file = export_training_data(
            days=export_days,
            min_confidence=0.8,
            output_dir="data/retraining"
        )
        
        # Generar reporte
        logger.info("📊 Generando reporte de exportación...")
        report = generate_export_report(output_file)
        
        # Registrar exportación
        record_export(
            export_type="retraining",
            records=report.get("total_samples", 0),
            file_path=output_file,
            metadata={
                "chart_type_distribution": report.get("chart_type_distribution", {}),
                "avg_confidence": report.get("avg_confidence", 0.0),
                "file_size_mb": report.get("file_size_mb", 0.0),
                "sources": report.get("sources", {}),
                "days_exported": export_days
            }
        )
        
        # Registrar métrica de exportación
        latency_ms = int((time.time() - start_time) * 1000)
        
        try:
            track_hybrid_execution(
                query="retraining_export",
                success=True,
                latency_ms=latency_ms,
                metadata={
                    "component": "retraining_export",
                    "records_exported": report.get("total_samples", 0),
                    "file_size_mb": report.get("file_size_mb", 0.0),
                    "chart_type_distribution": report.get("chart_type_distribution", {}),
                    "avg_confidence": report.get("avg_confidence", 0.0)
                }
            )
        except Exception as e:
            logger.warning(f"⚠️  Error registrando métrica: {e}")
        
        # Logging de reporte
        logger.info("=" * 60)
        logger.info("📊 REPORTE DE EXPORTACIÓN")
        logger.info("=" * 60)
        logger.info(f"✅ Total de ejemplos exportados: {report.get('total_samples', 0)}")
        logger.info(f"📁 Archivo: {output_file}")
        logger.info(f"💾 Tamaño: {report.get('file_size_mb', 0.0):.2f} MB")
        logger.info(f"📊 Confianza promedio: {report.get('avg_confidence', 0.0):.2%}")
        logger.info(f"⏱️  Tiempo de exportación: {latency_ms}ms")
        
        logger.info("\n📊 Distribución por tipo de gráfica:")
        for chart_type, count in sorted(
            report.get("chart_type_distribution", {}).items(),
            key=lambda x: -x[1]
        ):
            logger.info(f"   {chart_type}: {count}")
        
        logger.info("\n📊 Fuentes de datos:")
        sources = report.get("sources", {})
        logger.info(f"   Performance Metrics: {sources.get('performance_metrics', 0)}")
        logger.info(f"   User Feedback: {sources.get('feedback', 0)}")
        
        # Cleanup
        logger.info("\n🧹 Ejecutando cleanup de archivos antiguos...")
        deleted_count = cleanup_old_exports()
        if deleted_count > 0:
            logger.info(f"🗑️  Eliminados {deleted_count} archivos antiguos")
        else:
            logger.info("✅ No hay archivos antiguos para eliminar")
        
        logger.info("=" * 60)
        logger.info("✅ Exportación completada exitosamente")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ERROR EN EXPORTACIÓN")
        logger.error("=" * 60)
        logger.error(f"❌ Error: {e}", exc_info=True)
        
        # Registrar métrica de error
        try:
            latency_ms = int((time.time() - start_time) * 1000)
            track_hybrid_execution(
                query="retraining_export",
                success=False,
                latency_ms=latency_ms,
                error_message=str(e)
            )
        except:
            pass
        
        return False


# ============ Main ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de monitoreo semanal para exportación de datos de reentrenamiento"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar exportación sin verificar fecha"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular exportación sin ejecutar realmente"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Días hacia atrás para exportar (default: {RETRAINING_EXPORT_INTERVAL_DAYS})"
    )
    
    args = parser.parse_args()
    
    try:
        success = check_and_export(
            force=args.force,
            dry_run=args.dry_run,
            days=args.days
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Exportación cancelada por el usuario")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
        sys.exit(1)

