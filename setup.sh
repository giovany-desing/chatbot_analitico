#!/bin/bash

# Script de setup para Chatbot Analítico
# Verifica requisitos y levanta todo el sistema

set -e  # Salir si algún comando falla

echo "🚀 Chatbot Analítico - Setup Script"
echo "===================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir en color
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo "ℹ️  $1"
}

# 1. Verificar Docker
echo "1️⃣  Verificando Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker no está instalado"
    echo "   Instala Docker desde: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker instalado: $(docker --version)"

# 2. Verificar Docker Compose
echo ""
echo "2️⃣  Verificando Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose no está instalado"
    echo "   Instala Docker Compose desde: https://docs.docker.com/compose/install/"
    exit 1
fi
print_success "Docker Compose instalado: $(docker-compose --version)"

# 3. Verificar archivo .env
echo ""
echo "3️⃣  Verificando archivo .env..."
if [ ! -f .env ]; then
    print_warning ".env no existe, creando desde .env.example..."

    if [ -f .env.example ]; then
        cp .env.example .env
        print_success ".env creado desde .env.example"
        print_warning "IMPORTANTE: Edita .env con tus credenciales reales"
        echo ""
        echo "   Debes configurar:"
        echo "   - GROQ_API_KEY"
        echo "   - MYSQL_HOST"
        echo "   - MYSQL_USER"
        echo "   - MYSQL_PASSWORD"
        echo "   - MYSQL_DATABASE"
        echo ""
        read -p "¿Ya configuraste el archivo .env? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Configura .env y ejecuta este script de nuevo"
            exit 1
        fi
    else
        print_error ".env.example no existe"
        exit 1
    fi
else
    print_success ".env existe"

    # Verificar que tenga las variables críticas
    if grep -q "your_groq_api_key_here" .env || grep -q "gsk_your" .env; then
        print_warning ".env parece tener valores de ejemplo"
        echo ""
        read -p "¿Estás seguro que configuraste las credenciales reales? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Configura .env con credenciales reales"
            exit 1
        fi
    fi
fi

# 4. Verificar puertos disponibles
echo ""
echo "4️⃣  Verificando puertos disponibles..."

check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Puerto $port ($service) está en uso"
        echo "   PID: $(lsof -Pi :$port -sTCP:LISTEN -t)"
        return 1
    else
        print_success "Puerto $port ($service) disponible"
        return 0
    fi
}

PORTS_OK=true
check_port 8000 "API" || PORTS_OK=false
check_port 8501 "Frontend" || PORTS_OK=false
check_port 6379 "Redis" || PORTS_OK=false
check_port 5432 "PostgreSQL" || PORTS_OK=false

if [ "$PORTS_OK" = false ]; then
    echo ""
    read -p "¿Continuar de todos modos? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Libera los puertos y ejecuta el script de nuevo"
        exit 1
    fi
fi

# 5. Construir imágenes
echo ""
echo "5️⃣  Construyendo imágenes Docker..."
print_info "Esto puede tardar varios minutos la primera vez..."
if docker-compose build; then
    print_success "Imágenes construidas correctamente"
else
    print_error "Error construyendo imágenes"
    exit 1
fi

# 6. Iniciar servicios
echo ""
echo "6️⃣  Iniciando servicios..."
if docker-compose up -d; then
    print_success "Servicios iniciados"
else
    print_error "Error iniciando servicios"
    exit 1
fi

# 7. Esperar a que los servicios estén listos
echo ""
echo "7️⃣  Esperando a que los servicios estén listos..."

wait_for_service() {
    local service=$1
    local url=$2
    local max_wait=$3
    local waited=0

    echo -n "   Esperando $service..."
    while [ $waited -lt $max_wait ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo ""
            print_success "$service está listo"
            return 0
        fi
        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done
    echo ""
    print_error "$service no respondió en ${max_wait}s"
    return 1
fi

# Esperar API (60s)
wait_for_service "API" "http://localhost:8000/health" 60

# Esperar Frontend (30s)
wait_for_service "Frontend" "http://localhost:8501" 30

# 8. Verificar health check
echo ""
echo "8️⃣  Verificando estado de la API..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)

if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    print_success "API está healthy"

    # Mostrar estado de bases de datos
    echo ""
    echo "   Estado de bases de datos:"
    if echo "$HEALTH_RESPONSE" | grep -q '"mysql":true'; then
        print_success "   MySQL: Conectado"
    else
        print_error "   MySQL: Desconectado"
    fi

    if echo "$HEALTH_RESPONSE" | grep -q '"redis":true'; then
        print_success "   Redis: Conectado"
    else
        print_error "   Redis: Desconectado"
    fi

    if echo "$HEALTH_RESPONSE" | grep -q '"postgres":true'; then
        print_success "   PostgreSQL: Conectado"
    else
        print_error "   PostgreSQL: Desconectado"
    fi
else
    print_warning "API está en estado 'degraded'"
    echo "   Algunas bases de datos pueden no estar conectadas"
fi

# 9. Resumen final
echo ""
echo "========================================="
print_success "¡Setup completado!"
echo "========================================="
echo ""
echo "🌐 URLs disponibles:"
echo ""
echo "   Frontend (Streamlit):  http://localhost:8501"
echo "   API (FastAPI):         http://localhost:8000"
echo "   API Docs (Swagger):    http://localhost:8000/docs"
echo ""
echo "📋 Comandos útiles:"
echo ""
echo "   Ver logs:              docker-compose logs -f"
echo "   Ver logs del frontend: docker-compose logs -f frontend"
echo "   Ver logs de la API:    docker-compose logs -f app"
echo "   Detener servicios:     docker-compose stop"
echo "   Reiniciar servicios:   docker-compose restart"
echo "   Ver estado:            docker-compose ps"
echo ""
echo "🧪 Prueba el chatbot:"
echo ""
echo "   1. Abre http://localhost:8501 en tu navegador"
echo "   2. Verifica que diga '✅ API conectada' en el sidebar"
echo "   3. Haz una pregunta como: '¿Cuántas ventas hay?'"
echo ""
print_info "¡Disfruta del chatbot! 🎉"
