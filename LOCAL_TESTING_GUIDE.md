# 🏠 Guía de Testing Local con Configuración AWS

Esta guía te permite probar **localmente** la misma configuración que está desplegada en **AWS**, usando Docker Compose.

## 🎯 ¿Qué Incluye?

### Servicios Locales (Docker):
- ✅ **PostgreSQL + pgvector** - Base de datos RAG local
- ✅ **Redis** - Cache local
- ✅ **FastAPI App** - Aplicación principal

### Servicios AWS (Externos):
- 🌐 **MySQL RDS** - Base de datos de ventas textiles (ya desplegado en AWS)
- 🌐 **Fine-tuned Model** - Modelo en Modal.com
- 🌐 **Groq API** - LLM para procesamiento

---

## 📋 Paso 1: Configurar Variables de Entorno

### 1.1 Crear archivo `.env.local`

```bash
cd ~/Desktop/chatbot_analitico
cp .env.local.template .env.local
```

### 1.2 Obtener valores desde AWS Parameter Store

Ejecuta estos comandos para obtener los valores reales desde AWS:

```bash
# Groq API Key
aws ssm get-parameter \
  --name "/chatbot-analitico/prod/api/groq_key" \
  --region us-east-1 \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text

# Modal API Key
aws ssm get-parameter \
  --name "/chatbot-analitico/prod/api/modal_key" \
  --region us-east-1 \
  --with-decryption \
  --query 'Parameter.Value' \
  --output text

# Fine-tuned Model Endpoint
aws ssm get-parameter \
  --name "/chatbot-analitico/prod/api/finetuned_model_endpoint" \
  --region us-east-1 \
  --query 'Parameter.Value' \
  --output text
```

### 1.3 Editar `.env.local`

Abre `.env.local` y reemplaza los valores `TODO` con los valores reales obtenidos arriba:

```bash
nano .env.local
# O usa tu editor favorito: code .env.local, vim .env.local, etc.
```

Ejemplo de `.env.local` completo:

```env
# PostgreSQL Local
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=chatbot_user
POSTGRES_PASSWORD=chatbot_pass_local_123
POSTGRES_DB=chatbot_rag

# MySQL RDS AWS (ya configurado)
MYSQL_HOST=textil.cof2oucystyr.us-east-1.rds.amazonaws.com
MYSQL_PORT=3306
MYSQL_USER=samaca
MYSQL_PASSWORD=Mirringa2024New!
MYSQL_DATABASE=textil

# Redis Local
REDIS_URL=redis://redis:6379/0
REDIS_TTL=3600

# API Keys (valores reales)
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
MODAL_API_KEY=ak-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
FINETUNED_MODEL_ENDPOINT=https://XXXXXXXX--viz-model-fastapi-app.modal.run

# LLM Config
LLM_MODEL=llama-3.1-70b-versatile
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2000
LLM_TIMEOUT=30

# App Config
APP_NAME=ChatbotAnalitico
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
AWS_DEFAULT_REGION=us-east-1
```

---

## 🚀 Paso 2: Iniciar Servicios Locales

### Opción A: Usar el script automatizado (Recomendado)

```bash
./run_local.sh
```

Este script:
1. Verifica que `.env.local` exista
2. Detiene contenedores anteriores
3. Construye e inicia todos los servicios
4. Muestra logs y URLs útiles

### Opción B: Usar Docker Compose directamente

```bash
# Cargar variables de entorno
export $(grep -v '^#' .env.local | xargs)

# Iniciar servicios
docker-compose -f docker-compose.local.yml up -d --build

# Ver logs
docker-compose -f docker-compose.local.yml logs -f app
```

---

## 🧪 Paso 3: Probar la Aplicación

### 3.1 Verificar Health Check

```bash
curl http://localhost:8000/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "database": "ok",
  "redis": "ok"
}
```

### 3.2 Probar el Chatbot

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cuáles son las ventas totales del último mes?"
  }'
```

### 3.3 Ver Documentación Interactiva

Abre en tu navegador:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📊 Comandos Útiles

### Ver logs en tiempo real

```bash
# Logs de la app
docker-compose -f docker-compose.local.yml logs -f app

# Logs de todos los servicios
docker-compose -f docker-compose.local.yml logs -f

# Logs de PostgreSQL
docker-compose -f docker-compose.local.yml logs -f postgres

# Logs de Redis
docker-compose -f docker-compose.local.yml logs -f redis
```

### Verificar estado de contenedores

```bash
docker-compose -f docker-compose.local.yml ps
```

### Reiniciar servicios

```bash
# Reiniciar solo la app
docker-compose -f docker-compose.local.yml restart app

# Reiniciar todos los servicios
docker-compose -f docker-compose.local.yml restart

# Reconstruir app (si cambiaste código)
docker-compose -f docker-compose.local.yml up -d --build app
```

### Conectarse a las bases de datos

```bash
# PostgreSQL
docker-compose -f docker-compose.local.yml exec postgres psql -U chatbot_user -d chatbot_rag

# Redis
docker-compose -f docker-compose.local.yml exec redis redis-cli

# Verificar conexión a MySQL RDS
docker-compose -f docker-compose.local.yml exec app python -c "
from app.db.connections import get_mysql
mysql = get_mysql()
print('✅ MySQL conectado correctamente')
"
```

### Entrar al contenedor

```bash
docker-compose -f docker-compose.local.yml exec app bash
```

### Ejecutar tests

```bash
# Ejecutar todos los tests
docker-compose -f docker-compose.local.yml exec app pytest

# Ejecutar tests específicos
docker-compose -f docker-compose.local.yml exec app pytest tests/test_agents.py

# Ejecutar con coverage
docker-compose -f docker-compose.local.yml exec app pytest --cov=app tests/
```

### Ejecutar scripts

```bash
# Indexar ejemplos del router
docker-compose -f docker-compose.local.yml exec app python scripts/index_router_examples.py

# Exportar datos de entrenamiento
docker-compose -f docker-compose.local.yml exec app python scripts/auto_export_training_data.py --days 7

# Test de integraciones
docker-compose -f docker-compose.local.yml exec app python scripts/test_finetuned_integration.py
```

---

## 🛑 Detener Servicios

### Detener contenedores (mantener datos)

```bash
docker-compose -f docker-compose.local.yml down
```

### Detener y eliminar volúmenes (eliminar datos)

```bash
docker-compose -f docker-compose.local.yml down -v
```

---

## 🔧 Troubleshooting

### Error: "Cannot connect to MySQL"

**Causa**: Tu IP local no tiene acceso al RDS MySQL en AWS.

**Solución**:
1. Verifica el Security Group del MySQL RDS
2. Agrega tu IP pública actual:
   ```bash
   # Obtener tu IP pública
   curl -s https://ifconfig.me

   # Agregar regla al Security Group
   aws ec2 authorize-security-group-ingress \
     --group-id sg-044a5ea0a2b52643b \
     --protocol tcp \
     --port 3306 \
     --cidr $(curl -s https://ifconfig.me)/32
   ```

### Error: "Cannot connect to Fine-tuned Model"

**Causa**: El endpoint de Modal.com está inactivo o mal configurado.

**Solución**:
1. Verifica que el endpoint esté correcto en `.env.local`
2. Prueba el endpoint manualmente:
   ```bash
   curl https://tu-endpoint--viz-model-fastapi-app.modal.run/health
   ```

### Error: "Port 5432 already in use"

**Causa**: Ya tienes PostgreSQL corriendo localmente.

**Solución**:
```bash
# Opción 1: Detener PostgreSQL local
brew services stop postgresql

# Opción 2: Cambiar el puerto en docker-compose.local.yml
# Edita la línea:
# ports:
#   - "5433:5432"  # Cambiar 5432 a 5433
```

### Error: "Redis connection failed"

**Causa**: Redis está corriendo pero la app no puede conectarse.

**Solución**:
```bash
# Verificar que Redis esté corriendo
docker-compose -f docker-compose.local.yml exec redis redis-cli ping

# Reiniciar Redis
docker-compose -f docker-compose.local.yml restart redis
```

### Los cambios en el código no se reflejan

**Causa**: El código está montado como volumen, pero Python cacheó módulos.

**Solución**:
```bash
# Reiniciar la app
docker-compose -f docker-compose.local.yml restart app

# O reconstruir la imagen si cambiaste requirements.txt
docker-compose -f docker-compose.local.yml up -d --build app
```

---

## 📝 Diferencias entre Local y AWS

| Componente | Local | AWS |
|-----------|-------|-----|
| PostgreSQL | Docker local | RDS PostgreSQL |
| Redis | Docker local | Contenedor en EC2 |
| MySQL | RDS AWS (mismo) | RDS AWS |
| Fine-tuned Model | Modal.com (mismo) | Modal.com |
| Groq API | API Externa (mismo) | API Externa |
| S3 Buckets | Opcional | Activo |
| Logs | Docker logs | CloudWatch |
| Métricas | Local PostgreSQL | RDS PostgreSQL |

---

## 🎨 Desarrollo con Hot Reload

Los siguientes directorios están montados como volúmenes, por lo que **los cambios se reflejan automáticamente**:

- `./app` → Código de la aplicación
- `./data` → Datos y ejemplos
- `./scripts` → Scripts de utilidad
- `./tests` → Tests

**Ejemplo de workflow de desarrollo**:

1. Edita `app/agents/nodes.py`
2. Guarda el archivo
3. Reinicia la app: `docker-compose -f docker-compose.local.yml restart app`
4. Los cambios están activos inmediatamente

---

## 🚀 Workflow Completo: Local → AWS

### 1. Desarrollar y Probar Localmente

```bash
# Iniciar servicios locales
./run_local.sh

# Hacer cambios en el código
code app/

# Probar cambios
docker-compose -f docker-compose.local.yml restart app
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "test"}'

# Ejecutar tests
docker-compose -f docker-compose.local.yml exec app pytest
```

### 2. Desplegar a AWS

```bash
# Detener servicios locales
docker-compose -f docker-compose.local.yml down

# Desplegar a AWS
./deploy_to_ec2.sh

# Verificar en AWS
curl http://44.213.129.192:8000/health
```

---

## ✅ Checklist de Verificación

Antes de considerar que todo funciona correctamente:

- [ ] Health check responde OK: `curl http://localhost:8000/health`
- [ ] PostgreSQL local conectado
- [ ] Redis local conectado
- [ ] MySQL RDS conectado (AWS)
- [ ] Fine-tuned model accesible
- [ ] Groq API funcional
- [ ] Chatbot responde queries: `/chat` endpoint funciona
- [ ] Métricas se guardan en PostgreSQL
- [ ] Tests pasan: `pytest`

---

## 🎯 Resumen

**Ventajas de este setup**:
1. ✅ Misma configuración que AWS (excepto PostgreSQL y Redis que son locales)
2. ✅ No necesitas pagar por RDS PostgreSQL durante desarrollo
3. ✅ Puedes probar cambios rápidamente antes de desplegar
4. ✅ Hot reload para desarrollo rápido
5. ✅ Mismo código, mismas variables de entorno
6. ✅ Fácil debugging con logs y acceso directo a contenedores

**Cuándo usar Local vs AWS**:
- **Local**: Desarrollo, testing, debugging, pruebas rápidas
- **AWS**: Producción, testing de integración completo, performance real

---

¿Tienes preguntas? Revisa la sección de **Troubleshooting** o ejecuta `./run_local.sh` y revisa los logs.
