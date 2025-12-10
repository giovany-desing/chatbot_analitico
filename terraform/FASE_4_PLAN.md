# 🚀 FASE 4: Monitoreo, Optimización y CI/CD

## 📊 Objetivos de esta Fase

1. **Monitoreo Completo** - Visibilidad total del sistema en producción
2. **Alertas Automáticas** - Notificaciones proactivas de problemas
3. **Backups Automáticos** - Protección de datos críticos
4. **CI/CD Pipeline** - Despliegues automatizados y seguros
5. **Optimización de Costos** - Mantener costos en Free Tier
6. **Security Hardening** - Rotación de secrets y auditoría

---

## 📋 PASO 1: CloudWatch Logs y Monitoreo

### Objetivo
Capturar todos los logs de la aplicación en CloudWatch para debugging y análisis.

### Componentes
- **CloudWatch Log Groups** - Almacenar logs de contenedores
- **CloudWatch Agent** - Enviar logs desde EC2
- **Log Insights** - Queries sobre logs

### Archivos a Crear
1. `terraform/modules/monitoring/main.tf` - Infraestructura de monitoreo
2. `terraform/modules/monitoring/outputs.tf` - Outputs del módulo
3. `terraform/modules/monitoring/variables.tf` - Variables del módulo
4. `cloudwatch-agent-config.json` - Configuración del agente

### Métricas a Monitorear
- ✅ CPU/Memory de EC2
- ✅ Conexiones a RDS
- ✅ Latencia de queries
- ✅ Errores de aplicación
- ✅ Uso de Redis

---

## 📋 PASO 2: Dashboard de Métricas

### Objetivo
Crear un dashboard visual en CloudWatch con todas las métricas clave.

### Widgets del Dashboard
1. **Salud del Sistema**
   - EC2 CPU/Memory
   - RDS connections
   - Redis hit rate

2. **Performance de Aplicación**
   - Latencia de requests
   - Throughput (requests/min)
   - Error rate

3. **Base de Datos**
   - RDS CPU
   - RDS connections count
   - Slow queries

4. **Costos**
   - Estimated charges
   - Resource usage

### Archivo a Crear
- `terraform/modules/monitoring/dashboard.tf` - Dashboard CloudWatch

---

## 📋 PASO 3: Alarmas CloudWatch

### Objetivo
Recibir notificaciones automáticas cuando algo sale mal.

### Alarmas a Configurar

#### 1. Aplicación
- ❌ **High Error Rate** - Más de 5 errores en 5 minutos
- ❌ **Application Down** - Health check falla 3 veces consecutivas
- ❌ **High Response Time** - Latencia > 2 segundos

#### 2. Infraestructura
- ❌ **High CPU** - EC2 CPU > 80% por 5 minutos
- ❌ **High Memory** - Memory > 85%
- ❌ **Disk Space Low** - Disk usage > 90%

#### 3. Base de Datos
- ❌ **RDS High CPU** - CPU > 80%
- ❌ **RDS High Connections** - Connections > 80% del máximo
- ❌ **RDS Storage Low** - Storage < 10% libre

#### 4. Costos
- ❌ **Budget Alert** - Costos > $30/mes

### Archivos a Crear
- `terraform/modules/monitoring/alarms.tf` - Alarmas CloudWatch
- `terraform/modules/monitoring/sns.tf` - SNS topic para notificaciones

---

## 📋 PASO 4: Backups Automáticos

### Objetivo
Asegurar que los datos críticos tengan backup automático.

### Componentes

#### 1. RDS Backups (Ya configurado)
- ✅ Backup retention: 1 día (Free Tier)
- ✅ Backup window: 03:00-04:00 UTC
- 🔄 **Mejorar**: Copiar snapshots a S3 semanalmente

#### 2. PostgreSQL Data Export
- 📊 Exportar métricas de performance a S3
- 📊 Exportar datos de entrenamiento a S3
- 🔄 Frecuencia: Semanal

#### 3. Configuración Backup
- ⚙️ Terraform state (ya en S3)
- ⚙️ Parameter Store values
- ⚙️ Secrets backup encriptado

### Archivos a Crear
- `terraform/modules/backup/main.tf` - Backup automation
- `scripts/backup_rds_to_s3.sh` - Script de backup manual
- `scripts/restore_from_backup.sh` - Script de restore

---

## 📋 PASO 5: Rotación de Secrets

### Objetivo
Rotar contraseñas automáticamente para mayor seguridad.

### Secrets a Rotar
1. **RDS PostgreSQL Password** - Cada 90 días
2. **MySQL Password** - Coordinado con equipo externo
3. **API Keys** - Según políticas de proveedores

### Componentes
- **AWS Secrets Manager** - Almacenar secrets con rotación
- **Lambda Function** - Rotar passwords automáticamente
- **EventBridge Rule** - Trigger de rotación

### Archivos a Crear
- `terraform/modules/secrets_rotation/main.tf` - Infraestructura
- `terraform/modules/secrets_rotation/lambda.py` - Función de rotación

### Notas
⚠️ **Secrets Manager NO está en Free Tier** ($0.40/secret/mes)
- **Alternativa**: Mantener Parameter Store y rotar manualmente cada trimestre
- **Recomendación**: Implementar solo si el proyecto va a producción real

---

## 📋 PASO 6: CI/CD Pipeline

### Objetivo
Automatizar testing y deployment con GitHub Actions.

### Pipeline Stages

#### 1. **Pull Request** (Automatizado)
```
[PR Created] → Run Tests → Code Quality → Security Scan → ✅ Approve
```

#### 2. **Merge to Main** (Automatizado)
```
[Merge] → Build Docker → Push to ECR → Deploy to EC2 → Health Check → ✅ Done
```

#### 3. **Rollback** (Manual)
```
[Issue Detected] → Trigger Rollback → Deploy Previous Version → ✅ Recovered
```

### Componentes

#### Tests Automáticos
- ✅ Unit tests (pytest)
- ✅ Integration tests
- ✅ Security scan (Bandit)
- ✅ Linting (flake8, black)

#### Build
- 🐳 Build Docker image
- 📦 Tag with commit SHA
- ☁️ Push to Amazon ECR

#### Deploy
- 🚀 SSH to EC2
- 🔄 Pull new image
- 🔄 Rolling restart
- ✅ Health check verification

### Archivos a Crear
- `.github/workflows/ci.yml` - CI pipeline (tests)
- `.github/workflows/deploy.yml` - CD pipeline (deploy)
- `scripts/health_check.sh` - Verificación post-deploy
- `scripts/rollback.sh` - Rollback automático

---

## 📋 PASO 7: Optimización de Costos

### Objetivo
Mantener costos dentro del Free Tier (~$0-5/mes).

### Análisis Actual

| Recurso | Costo Estimado | Free Tier | Exceso |
|---------|---------------|-----------|--------|
| EC2 t3.micro | $0 | ✅ 750 hrs/mes | No |
| RDS db.t3.micro | $0 | ✅ 750 hrs/mes | No |
| EBS 16GB | $0 | ✅ 30GB/mes | No |
| S3 Storage | $0.02/mes | ✅ 5GB/mes | No |
| Data Transfer | $0-2/mes | ⚠️ 100GB/mes out | Posible |
| CloudWatch Logs | $0-3/mes | ⚠️ 5GB/mes | Posible |
| **Total Estimado** | **$2-5/mes** | | |

### Optimizaciones

#### 1. CloudWatch Logs
- ✅ Configurar retention: 7 días (en vez de indefinido)
- ✅ Filtrar logs: Solo ERROR y WARNING
- ✅ Usar log sampling para DEBUG

#### 2. S3
- ✅ Lifecycle policies: Mover a Glacier después de 30 días
- ✅ Limpiar training data antigua automáticamente

#### 3. Data Transfer
- ✅ Usar CloudFront CDN (Free Tier: 1TB/mes)
- ✅ Comprimir responses

#### 4. RDS
- ✅ Mantener backup retention en 1 día
- ✅ Desactivar Multi-AZ (no es Free Tier)

### Archivos a Crear
- `terraform/modules/cost_optimization/main.tf` - Políticas de optimización
- `scripts/cleanup_old_data.sh` - Limpieza de datos antiguos

---

## 📋 PASO 8: Budget Alerts

### Objetivo
Recibir alertas si los costos exceden el presupuesto.

### Budgets a Configurar

#### 1. Monthly Budget
- **Monto**: $25/mes
- **Alertas**:
  - 50% ($12.50) - Email warning
  - 80% ($20) - Email critical
  - 100% ($25) - Email + SMS

#### 2. Service-Specific Budgets
- **CloudWatch**: $5/mes
- **S3**: $2/mes
- **Data Transfer**: $3/mes

### Archivo a Crear
- `terraform/modules/billing/main.tf` - AWS Budgets

---

## 🗂️ Estructura de Archivos Final

```
terraform/
├── modules/
│   ├── monitoring/
│   │   ├── main.tf           # CloudWatch Logs, Agent
│   │   ├── dashboard.tf      # CloudWatch Dashboard
│   │   ├── alarms.tf         # CloudWatch Alarms
│   │   ├── sns.tf            # SNS Topics
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── backup/
│   │   ├── main.tf           # Backup automation
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── cost_optimization/
│   │   ├── main.tf           # Lifecycle policies, retention
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── billing/
│       ├── main.tf           # AWS Budgets
│       ├── variables.tf
│       └── outputs.tf
├── main.tf                   # Actualizar con nuevos módulos
└── FASE_4_PLAN.md           # Este archivo

.github/
└── workflows/
    ├── ci.yml                # CI pipeline
    └── deploy.yml            # CD pipeline

scripts/
├── backup_rds_to_s3.sh      # Backup manual
├── restore_from_backup.sh   # Restore manual
├── health_check.sh          # Health check post-deploy
├── rollback.sh              # Rollback automático
└── cleanup_old_data.sh      # Limpieza de datos

docs/
└── cloudwatch-agent-config.json  # Configuración CloudWatch Agent
```

---

## 📝 Orden de Implementación Recomendado

### Semana 1: Monitoreo Básico
1. ✅ CloudWatch Logs (PASO 1)
2. ✅ Dashboard básico (PASO 2)
3. ✅ Alarmas críticas (PASO 3)

### Semana 2: Backups y Seguridad
4. ✅ Backups automáticos (PASO 4)
5. ⏭️ Secrets rotation (PASO 5) - Opcional

### Semana 3: CI/CD
6. ✅ GitHub Actions CI (PASO 6)
7. ✅ GitHub Actions CD (PASO 6)

### Semana 4: Optimización
8. ✅ Cost optimization (PASO 7)
9. ✅ Budget alerts (PASO 8)

---

## ⚙️ Decisiones de Arquitectura

### ¿Usar Secrets Manager o Parameter Store?
- **Secrets Manager**: $0.40/secret/mes + rotación automática
- **Parameter Store**: Gratis + rotación manual
- **Decisión**: **Parameter Store** (proyecto personal, Free Tier)

### ¿Usar ECR o Docker Hub?
- **ECR**: 500MB gratis/mes, después $0.10/GB
- **Docker Hub**: Gratis ilimitado (público)
- **Decisión**: **Docker Hub** (para Free Tier)

### ¿CloudWatch o Third-Party Monitoring?
- **CloudWatch**: Integrado, $0.30/métrica custom
- **Datadog/New Relic**: $15-30/mes mínimo
- **Decisión**: **CloudWatch** (Free Tier parcial)

### ¿GitHub Actions o AWS CodePipeline?
- **GitHub Actions**: 2000 min/mes gratis
- **CodePipeline**: $1/pipeline activo
- **Decisión**: **GitHub Actions** (más features, gratis)

---

## 🎯 Métricas de Éxito de FASE 4

Al finalizar la FASE 4, deberías tener:

- ✅ **Visibilidad Total**: Dashboard con todas las métricas
- ✅ **Alertas Configuradas**: Notificaciones automáticas de problemas
- ✅ **Backups Funcionando**: Datos protegidos y recuperables
- ✅ **CI/CD Operativo**: Deploy con un click
- ✅ **Costos Controlados**: $0-5/mes, alertas configuradas
- ✅ **Documentación Completa**: Runbooks y procedimientos

---

## 🚀 ¿Listo para Empezar?

Podemos implementar la FASE 4 de dos formas:

### Opción A: Implementación Completa (Recomendado)
Implementar todos los pasos en orden, con enfoque enterprise completo.

### Opción B: Implementación Mínima (Free Tier Focus)
Solo los componentes esenciales que no generan costos:
- CloudWatch Logs básico
- Alarmas críticas gratis
- GitHub Actions CI/CD
- Budget alerts

**¿Qué opción prefieres?**
