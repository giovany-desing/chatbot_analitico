# AWS Systems Manager Parameter Store para credenciales
# Servicio gratuito (10,000 parámetros incluidos en Free Tier)

# Database credentials
resource "aws_ssm_parameter" "db_host" {
  name        = "/${var.project_name}/${var.environment}/db/host"
  description = "PostgreSQL RDS endpoint"
  type        = "String"
  value       = var.db_endpoint

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-host"
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "db_name" {
  name        = "/${var.project_name}/${var.environment}/db/name"
  description = "PostgreSQL database name"
  type        = "String"
  value       = var.db_name

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-name"
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "db_username" {
  name        = "/${var.project_name}/${var.environment}/db/username"
  description = "PostgreSQL master username"
  type        = "SecureString"  # Encriptado con KMS
  value       = var.db_username

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-username"
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "db_password" {
  name        = "/${var.project_name}/${var.environment}/db/password"
  description = "PostgreSQL master password"
  type        = "SecureString"  # Encriptado con KMS
  value       = var.db_password

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-password"
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "db_port" {
  name        = "/${var.project_name}/${var.environment}/db/port"
  description = "PostgreSQL port"
  type        = "String"
  value       = "5432"

  tags = {
    Name        = "${var.project_name}-${var.environment}-db-port"
    Environment = var.environment
  }
}

# API Keys
resource "aws_ssm_parameter" "groq_api_key" {
  name        = "/${var.project_name}/${var.environment}/api/groq_key"
  description = "Groq API Key"
  type        = "SecureString"
  value       = var.groq_api_key

  tags = {
    Name        = "${var.project_name}-${var.environment}-groq-api-key"
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "modal_api_key" {
  name        = "/${var.project_name}/${var.environment}/api/modal_key"
  description = "Modal API Key"
  type        = "SecureString"
  value       = var.modal_api_key

  tags = {
    Name        = "${var.project_name}-${var.environment}-modal-api-key"
    Environment = var.environment
  }
}

# External MySQL connection (if exists)
resource "aws_ssm_parameter" "external_mysql_uri" {
  name        = "/${var.project_name}/${var.environment}/external/mysql_uri"
  description = "External MySQL connection URI"
  type        = "SecureString"
  value       = var.external_mysql_uri

  tags = {
    Name        = "${var.project_name}-${var.environment}-mysql-uri"
    Environment = var.environment
  }
}

# S3 bucket names
resource "aws_ssm_parameter" "s3_training_bucket" {
  name        = "/${var.project_name}/${var.environment}/s3/training_bucket"
  description = "S3 bucket for training data"
  type        = "String"
  value       = var.s3_training_bucket

  tags = {
    Name        = "${var.project_name}-${var.environment}-s3-training"
    Environment = var.environment
  }
}

resource "aws_ssm_parameter" "s3_backups_bucket" {
  name        = "/${var.project_name}/${var.environment}/s3/backups_bucket"
  description = "S3 bucket for backups"
  type        = "String"
  value       = var.s3_backups_bucket

  tags = {
    Name        = "${var.project_name}-${var.environment}-s3-backups"
    Environment = var.environment
  }
}

