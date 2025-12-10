#!/bin/bash
set -e

EC2_IP="44.213.129.192"
EC2_USER="ec2-user"
SSH_KEY="$HOME/.ssh/chatbot-key.pem"

echo "🚀 Desplegando aplicación a EC2..."

# 1. Comprimir código
echo "📦 Comprimiendo código..."
tar -czf app_deploy.tar.gz \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='venv' \
  --exclude='terraform' \
  --exclude='backups' \
  --exclude='*.log' \
  --exclude='.env' \
  --exclude='app.tar.gz' \
  app/ scripts/ data/ migrations/ requirements.txt Dockerfile docker-compose.aws.yml

# 2. Copiar a EC2
echo "📤 Copiando a EC2..."
scp -i "$SSH_KEY" app_deploy.tar.gz "$EC2_USER@$EC2_IP:/tmp/"

# 3. Desplegar
echo "⚙️  Desplegando en EC2..."
ssh -i "$SSH_KEY" "$EC2_USER@$EC2_IP" << 'ENDSSH'
set -e
cd /opt/chatbot

# Detener contenedores
echo "🛑 Deteniendo contenedores..."
docker-compose -f docker-compose.aws.yml down 2>/dev/null || true

# Limpiar
echo "🧹 Limpiando..."
sudo rm -rf app/ scripts/ data/ migrations/ requirements.txt Dockerfile docker-compose.aws.yml .env

# Extraer
echo "📂 Extrayendo..."
sudo tar -xzf /tmp/app_deploy.tar.gz -C .
sudo chown -R ec2-user:ec2-user /opt/chatbot
rm /tmp/app_deploy.tar.gz

# Crear .env desde Parameter Store
echo "🔐 Obteniendo credenciales desde AWS SSM..."

# Obtener valores de Parameter Store
DB_HOST=$(aws ssm get-parameter --name "/chatbot-analitico/prod/db/host" --region us-east-1 --query 'Parameter.Value' --output text)
DB_NAME=$(aws ssm get-parameter --name "/chatbot-analitico/prod/db/name" --region us-east-1 --query 'Parameter.Value' --output text)
DB_USER=$(aws ssm get-parameter --name "/chatbot-analitico/prod/db/username" --region us-east-1 --with-decryption --query 'Parameter.Value' --output text)
DB_PASS=$(aws ssm get-parameter --name "/chatbot-analitico/prod/db/password" --region us-east-1 --with-decryption --query 'Parameter.Value' --output text)
GROQ_KEY=$(aws ssm get-parameter --name "/chatbot-analitico/prod/api/groq_key" --region us-east-1 --with-decryption --query 'Parameter.Value' --output text)
MODAL_KEY=$(aws ssm get-parameter --name "/chatbot-analitico/prod/api/modal_key" --region us-east-1 --with-decryption --query 'Parameter.Value' --output text)
FINETUNED_ENDPOINT=$(aws ssm get-parameter --name "/chatbot-analitico/prod/api/finetuned_model_endpoint" --region us-east-1 --query 'Parameter.Value' --output text 2>/dev/null || echo "")
S3_TRAINING=$(aws ssm get-parameter --name "/chatbot-analitico/prod/s3/training_bucket" --region us-east-1 --query 'Parameter.Value' --output text)
S3_BACKUPS=$(aws ssm get-parameter --name "/chatbot-analitico/prod/s3/backups_bucket" --region us-east-1 --query 'Parameter.Value' --output text)

# Obtener MySQL URI
MYSQL_URI=$(aws ssm get-parameter --name "/chatbot-analitico/prod/external/mysql_uri" --region us-east-1 --with-decryption --query 'Parameter.Value' --output text 2>/dev/null || echo "")

# Parsear MySQL URI (formato: mysql://user:pass@host:port/db)
if [ ! -z "$MYSQL_URI" ] && [ "$MYSQL_URI" != "NOT_CONFIGURED" ] && [ "$MYSQL_URI" != "PLACEHOLDER" ]; then
  MYSQL_USER=$(echo "$MYSQL_URI" | sed -n 's|.*://\([^:]*\):.*|\1|p')
  MYSQL_PASSWORD=$(echo "$MYSQL_URI" | sed -n 's|.*://[^:]*:\([^@]*\)@.*|\1|p')
  MYSQL_HOST=$(echo "$MYSQL_URI" | sed -n 's|.*@\([^:]*\):.*|\1|p')
  MYSQL_PORT=$(echo "$MYSQL_URI" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
  MYSQL_DATABASE=$(echo "$MYSQL_URI" | sed -n 's|.*/\([^?]*\).*|\1|p')
else
  MYSQL_USER=""
  MYSQL_PASSWORD=""
  MYSQL_HOST=""
  MYSQL_PORT="3306"
  MYSQL_DATABASE=""
fi

echo "📝 Creando archivo .env..."

cat > .env << EOF
# PostgreSQL RDS
POSTGRES_HOST=${DB_HOST}
POSTGRES_PORT=5432
POSTGRES_DB=${DB_NAME}
POSTGRES_USER=${DB_USER}
POSTGRES_PASSWORD=${DB_PASS}

# MySQL externo
MYSQL_HOST=${MYSQL_HOST}
MYSQL_PORT=${MYSQL_PORT:-3306}
MYSQL_USER=${MYSQL_USER}
MYSQL_PASSWORD=${MYSQL_PASSWORD}
MYSQL_DATABASE=${MYSQL_DATABASE}

# API Keys
GROQ_API_KEY=${GROQ_KEY}
MODAL_API_KEY=${MODAL_KEY}
FINETUNED_MODEL_ENDPOINT=${FINETUNED_ENDPOINT}
OPENAI_API_KEY=

# S3
S3_TRAINING_BUCKET=${S3_TRAINING}
S3_BACKUPS_BUCKET=${S3_BACKUPS}

# App Config
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
AWS_DEFAULT_REGION=us-east-1
APP_NAME=ChatbotAnalitico
APP_VERSION=1.0.0
LLM_MODEL=llama-3.1-70b-versatile
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2000
LLM_TIMEOUT=30
USAR_FINETUNED_MODEL=true
REDIS_URL=
REDIS_TTL=3600
EOF

echo "✅ Archivo .env creado correctamente"

# Construir e iniciar
echo "🐳 Construyendo e iniciando..."
docker-compose -f docker-compose.aws.yml up -d --build

echo "⏳ Esperando inicio..."
sleep 15

echo "🔍 Estado de contenedores:"
docker-compose -f docker-compose.aws.yml ps

echo "✅ Deployment completado"
ENDSSH

# Limpiar local
rm -f app_deploy.tar.gz

echo ""
echo "🎉 Deployment exitoso"
echo "🌐 URL: http://$EC2_IP:8000"
echo ""
echo "📋 Ver logs:"
echo "  ssh -i $SSH_KEY $EC2_USER@$EC2_IP"
echo "  cd /opt/chatbot && docker-compose -f docker-compose.aws.yml logs -f app"
