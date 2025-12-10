#!/bin/bash
# ====================================================================
# Script para ejecutar el chatbot localmente con configuración AWS
# ====================================================================

set -e

echo "🚀 Iniciando Chatbot Analítico en modo local con configuración AWS"
echo ""

# Verificar que existe .env.local
if [ ! -f .env.local ]; then
    echo "❌ Error: No se encontró .env.local"
    echo ""
    echo "📝 Pasos para crear .env.local:"
    echo "   1. cp .env.local.template .env.local"
    echo "   2. Edita .env.local y completa los valores reales"
    echo "   3. Ejecuta este script nuevamente"
    echo ""
    exit 1
fi

# Cargar .env.local
export $(grep -v '^#' .env.local | xargs)

# Verificar valores críticos
echo "🔍 Verificando configuración..."
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  ADVERTENCIA: GROQ_API_KEY no está configurado"
fi

if [ -z "$MODAL_API_KEY" ]; then
    echo "⚠️  ADVERTENCIA: MODAL_API_KEY no está configurado"
fi

if [ -z "$FINETUNED_MODEL_ENDPOINT" ]; then
    echo "⚠️  ADVERTENCIA: FINETUNED_MODEL_ENDPOINT no está configurado"
fi

echo ""
echo "📊 Configuración detectada:"
echo "   - PostgreSQL: Local (Docker)"
echo "   - Redis: Local (Docker)"
echo "   - MySQL: RDS AWS ($MYSQL_HOST)"
echo "   - Fine-tuned Model: ${FINETUNED_MODEL_ENDPOINT:-No configurado}"
echo ""

# Detener contenedores anteriores si existen
echo "🛑 Deteniendo contenedores anteriores (si existen)..."
docker-compose -f docker-compose.local.yml down 2>/dev/null || true

# Construir e iniciar servicios
echo ""
echo "🔨 Construyendo e iniciando servicios..."
docker-compose -f docker-compose.local.yml up -d --build

# Esperar a que los servicios estén listos
echo ""
echo "⏳ Esperando a que los servicios estén listos..."
sleep 10

# Verificar estado de los contenedores
echo ""
echo "📦 Estado de los contenedores:"
docker-compose -f docker-compose.local.yml ps

# Verificar logs de la aplicación
echo ""
echo "📋 Logs de la aplicación (últimas 20 líneas):"
docker-compose -f docker-compose.local.yml logs --tail=20 app

echo ""
echo "✅ Servicios iniciados exitosamente!"
echo ""
echo "🌐 URLs disponibles:"
echo "   - API: http://localhost:8000"
echo "   - Docs: http://localhost:8000/docs"
echo "   - Health: http://localhost:8000/health"
echo ""
echo "📊 Comandos útiles:"
echo "   - Ver logs en vivo: docker-compose -f docker-compose.local.yml logs -f app"
echo "   - Detener servicios: docker-compose -f docker-compose.local.yml down"
echo "   - Reiniciar app: docker-compose -f docker-compose.local.yml restart app"
echo "   - Ejecutar tests: docker-compose -f docker-compose.local.yml exec app pytest"
echo ""
echo "🔧 Debugging:"
echo "   - Conectar a PostgreSQL: docker-compose -f docker-compose.local.yml exec postgres psql -U chatbot_user -d chatbot_rag"
echo "   - Conectar a Redis: docker-compose -f docker-compose.local.yml exec redis redis-cli"
echo "   - Entrar al contenedor: docker-compose -f docker-compose.local.yml exec app bash"
echo ""
