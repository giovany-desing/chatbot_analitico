# 🧹 Guía de Limpieza del Proyecto

## ⚠️ NUNCA ELIMINAR (ESENCIALES)

### 1. Directorio `venv/` (1.4GB)
**NO ELIMINAR** - Contiene todas las dependencias de Python instaladas. Si lo eliminas, la aplicación no funcionará.

### 2. Directorio `app/` (388KB)
**NO ELIMINAR** - Código principal de la aplicación:
- `app/main.py` - Punto de entrada FastAPI
- `app/agents/` - Sistema de agentes LangGraph
- `app/db/` - Conexiones a bases de datos
- `app/intelligence/` - Sistema híbrido y fine-tuned model
- `app/llm/` - Modelos de lenguaje
- `app/tools/` - Herramientas SQL y visualización
- `app/metrics/` - Sistema de métricas y alertas

### 3. Directorio `terraform/` (649MB - PERO VER ABAJO)
**PARCIALMENTE ELIMINABLE** - Infraestructura AWS:
- ✅ MANTENER: `terraform/*.tf`, `terraform/modules/**/*.tf`
- ❌ ELIMINAR: `terraform/.terraform/` (archivos descargados que se pueden regenerar con `terraform init`)

### 4. Archivos de Configuración
**NO ELIMINAR**:
- `requirements.txt` - Dependencias de Python
- `docker-compose.yml` - Configuración Docker local
- `docker-compose.aws.yml` - Configuración Docker AWS
- `Dockerfile` - Imagen Docker de la aplicación
- `.env` (si existe) - Variables de entorno

### 5. Datos Esenciales
**NO ELIMINAR**:
- `data/router_examples.json` - Ejemplos para clasificación RAG
- `data/sql_examples.json` - Ejemplos SQL
- `migrations/` - Migraciones de base de datos

### 6. Scripts de Despliegue
**NO ELIMINAR**:
- `deploy_to_ec2.sh` - Script de despliegue a AWS
- `setup.sh` - Script de configuración inicial

---

## ✅ PUEDES ELIMINAR (SEGURO)

### 1. Backups (920KB)
```bash
rm -rf backups/
```
**Razón**: Ya está migrado a AWS, el backup local ya no es necesario. Si quieres conservarlo, puedes subirlo a S3.

### 2. Archivos de Training Data Antiguos
```bash
rm training_data.jsonl
rm training_data_complete.jsonl
```
**Razón**: Estos archivos (516KB total) ya fueron procesados. El sistema genera nuevos datos de entrenamiento con `scripts/auto_export_training_data.py`.

### 3. Cache de Terraform (600MB+ dentro de terraform/)
```bash
cd terraform
rm -rf .terraform/
```
**Razón**: Se regenera automáticamente con `terraform init`. **IMPORTANTE**: Mantén `terraform.tfstate` y `terraform.tfstate.backup` si existen localmente.

### 4. Archivos de Test Temporales
```bash
rm test_simple.py
rm test_modal_endpoint.py
```
**Razón**: Tests temporales, no son parte del test suite formal en `tests/`.

### 5. Cache de Coverage
```bash
rm .coverage
rm -rf .pytest_cache/
```
**Razón**: Cache de pytest y coverage que se regenera al ejecutar tests.

### 6. Python Cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```
**Razón**: Cache de Python que se regenera automáticamente.

### 7. Archivos Temporales de macOS
```bash
find . -name ".DS_Store" -delete
```
**Razón**: Archivos de sistema de macOS que no afectan la aplicación.

### 8. Archivo Comprimido
```bash
rm app.tar.gz
```
**Razón**: Archivo temporal de despliegue antiguo.

### 9. Frontend Antiguo (si no lo usas)
```bash
rm front_app.py
rm Dockerfile.frontend
```
**Razón**: Si no estás usando la interfaz Streamlit actualmente.

---

## 🤔 DECIDIR SEGÚN TU CASO

### 1. Directorio `tests/` (400KB)
**MANTENER SI**: Planeas ejecutar tests en local.
**ELIMINAR SI**: Solo ejecutas la aplicación y no desarrollas localmente.

### 2. Scripts de Utilidad (160KB)
**REVISAR**:
- ✅ MANTENER:
  - `scripts/auto_export_training_data.py` - Exporta datos para reentrenamiento
  - `scripts/index_router_examples.py` - Indexa ejemplos RAG
  - `scripts/init_postgres.sql` - Inicialización PostgreSQL

- ❌ ELIMINAR (si no los usas):
  - `scripts/benchmark_hybrid.py` - Benchmarking (solo desarrollo)
  - `scripts/compare_charts.py` - Comparación de gráficas (solo desarrollo)
  - `scripts/test_*.py` - Scripts de prueba temporales
  - `scripts/get_my_ip.py` - Utilidad que ya no necesitas

### 3. Documentación Extra
**REVISAR**:
- `NOTIFICATIONS_CONFIG.md` - Documentación de notificaciones
- `TEST_ENDPOINT.md` - Documentación de testing
- `README.md` - Documentación principal (recomendado mantener)

---

## 🚀 COMANDO DE LIMPIEZA RÁPIDA (SEGURO)

```bash
cd ~/Desktop/chatbot_analitico

# Eliminar backups
rm -rf backups/

# Eliminar training data antiguos
rm -f training_data.jsonl training_data_complete.jsonl

# Eliminar cache de Terraform (¡CUIDADO! No elimines .tfstate)
cd terraform && rm -rf .terraform/ && cd ..

# Eliminar tests temporales
rm -f test_simple.py test_modal_endpoint.py

# Eliminar cache de Python y pytest
rm -f .coverage
rm -rf .pytest_cache/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Eliminar archivos de macOS
find . -name ".DS_Store" -delete

# Eliminar comprimidos temporales
rm -f app.tar.gz

echo "✅ Limpieza completada"
```

**Ahorro estimado**: ~1.5GB (principalmente por terraform/.terraform/)

---

## 📊 VERIFICACIÓN POST-LIMPIEZA

Después de limpiar, verifica que la aplicación siga funcionando:

```bash
# Verificar estructura esencial
ls -la app/
ls -la terraform/*.tf
ls -la data/
ls -la requirements.txt

# Si eliminaste .terraform/, regenerarlo:
cd terraform
terraform init

# Verificar que Docker funcione
docker-compose -f docker-compose.yml ps
```

---

## 💾 BACKUP ANTES DE ELIMINAR

Si tienes dudas, crea un backup antes de eliminar:

```bash
# Backup específico de lo que vas a eliminar
mkdir ~/backup_chatbot_temp
cp -r backups/ ~/backup_chatbot_temp/
cp training_data*.jsonl ~/backup_chatbot_temp/
cp test_*.py ~/backup_chatbot_temp/

# Comprimir
cd ~
tar -czf backup_chatbot_temp.tar.gz backup_chatbot_temp/
rm -rf backup_chatbot_temp/

echo "Backup creado en ~/backup_chatbot_temp.tar.gz"
```

---

## 📝 RESUMEN

| Directorio/Archivo | Tamaño | Acción | Ahorro |
|-------------------|--------|--------|--------|
| `venv/` | 1.4GB | ❌ **NO TOCAR** | - |
| `terraform/.terraform/` | ~600MB | ✅ Eliminar (regenerable) | 600MB |
| `backups/` | 920KB | ✅ Eliminar (ya migrado) | 920KB |
| `training_data*.jsonl` | 516KB | ✅ Eliminar (regenerable) | 516KB |
| `.coverage`, `.pytest_cache` | 76KB | ✅ Eliminar (cache) | 76KB |
| `__pycache__/`, `*.pyc` | ~20KB | ✅ Eliminar (cache) | 20KB |
| `test_*.py` temporales | 16KB | ✅ Eliminar (temporales) | 16KB |
| `app.tar.gz` | 4KB | ✅ Eliminar (temporal) | 4KB |
| `.DS_Store` | 8KB | ✅ Eliminar (macOS) | 8KB |

**Total ahorro potencial**: ~1.5GB

---

## ⚠️ ADVERTENCIAS FINALES

1. **NUNCA elimines `venv/`** - Romperías todas las dependencias
2. **Guarda `terraform.tfstate`** si existe localmente (aunque ahora está en S3)
3. **No elimines `app/`, `data/`, `scripts/` sin revisar** - Son código funcional
4. **Haz backup si tienes dudas** - Mejor prevenir que lamentar
