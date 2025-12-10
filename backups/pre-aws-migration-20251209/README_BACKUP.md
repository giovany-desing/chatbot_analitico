  # Backup Pre-AWS Migration

  **Fecha:** $(date)
  **Proyecto:** Chatbot Analítico

  ## Contenido:

  1. **chatbot_rag_backup.sql** - Base de datos PostgreSQL completa
     - router_examples (30 ejemplos indexados)
     - performance_metrics
     - feedback
     - export_metadata

  2. **.env.backup** - Variables de entorno originales

  3. **docker-compose.yml.backup** - Configuración Docker original

  4. **app_source_code.tar.gz** - Código fuente completo

  5. **router_examples.json.backup** - Dataset de clasificación

  6. **training_data_complete.jsonl.backup** - 500 ejemplos de fine-tuning

  ## Cómo restaurar:

  ### Restaurar PostgreSQL:
  ```bash
  docker-compose up -d postgres
  docker-compose exec -T postgres psql -U postgres -c "DROP DATABASE IF EXISTS chatbot_rag;"
  docker-compose exec -T postgres psql -U postgres -c "CREATE DATABASE chatbot_rag;"
  docker-compose exec -T postgres psql -U postgres chatbot_rag < chatbot_rag_backup.sql

  Restaurar código:

  tar -xzf app_source_code.tar.gz
  cp .env.backup ../.env
  cp docker-compose.yml.backup ../docker-compose.yml

  Estado antes de migración:

  - Docker Compose local con 4 servicios (app, frontend, postgres, redis)
  - PostgreSQL local con pgvector
  - MySQL RDS externo (no respaldado, ya está en AWS)
  - Modelo fine-tuned en Modal.com (no afectado)

  EOF
