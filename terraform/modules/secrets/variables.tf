variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

# Database
variable "db_endpoint" {
  description = "RDS endpoint"
  type        = string
}

variable "db_name" {
  description = "Database name"
  type        = string
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# API Keys
variable "groq_api_key" {
  description = "Groq API Key"
  type        = string
  sensitive   = true
  default     = "PLACEHOLDER"  # Se actualizará manualmente después
}

variable "modal_api_key" {
  description = "Modal API Key"
  type        = string
  sensitive   = true
  default     = "PLACEHOLDER"  # Se actualizará manualmente después
}

# External connections
variable "external_mysql_uri" {
  description = "External MySQL URI"
  type        = string
  sensitive   = true
  default     = "PLACEHOLDER"  # Se actualizará manualmente después
}

# S3 buckets
variable "s3_training_bucket" {
  description = "S3 training bucket name"
  type        = string
}

variable "s3_backups_bucket" {
  description = "S3 backups bucket name"
  type        = string
}

