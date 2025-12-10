# RDS PostgreSQL Instance with pgvector support
resource "aws_db_instance" "postgres" {
  identifier = "${var.project_name}-${var.environment}-postgres"

  # Engine
  engine               = "postgres"
  engine_version       = "15"  # Usar la última versión de la familia 15.x
  instance_class       = "db.t3.micro"  # Free Tier eligible (750 hrs/mes)

  # Storage
  allocated_storage     = 20  # GB - Free Tier incluye 20GB
  storage_type          = "gp3"
  storage_encrypted     = true
  max_allocated_storage = 0  # Deshabilitar autoscaling para Free Tier

  # Database
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  port     = 5432

  # Network
  db_subnet_group_name   = var.db_subnet_group_name
  vpc_security_group_ids = [var.db_security_group_id]
  publicly_accessible    = false  # Solo accesible desde VPC

  # Backups
  backup_retention_period = 1  # Free Tier: máximo 1 día
  backup_window          = "03:00-04:00"  # UTC
  maintenance_window     = "mon:04:00-mon:05:00"

  # Monitoring
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  performance_insights_enabled    = false  # No incluido en Free Tier
  monitoring_interval             = 0  # Deshabilitar enhanced monitoring (costo extra)

  # High Availability
  multi_az = false  # Single AZ para Free Tier

  # Deletion Protection
  deletion_protection       = false  # Cambiar a true en producción
  skip_final_snapshot      = true   # Cambiar a false en producción
  final_snapshot_identifier = "${var.project_name}-${var.environment}-final-snapshot"

  # Maintenance
  auto_minor_version_upgrade = true
  apply_immediately         = false

  # Parameter Group para pgvector
  parameter_group_name = aws_db_parameter_group.postgres.name

  tags = {
    Name        = "${var.project_name}-${var.environment}-postgres"
    Environment = var.environment
    Backup      = "true"
  }
}

# Parameter Group para habilitar pgvector
resource "aws_db_parameter_group" "postgres" {
  name   = "${var.project_name}-${var.environment}-postgres-params"
  family = "postgres15"

  description = "Custom parameter group for PostgreSQL 15"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-postgres-params"
    Environment = var.environment
  }
}
