variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "chatbot_rag"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

variable "db_subnet_group_name" {
  description = "DB subnet group name from networking module"
  type        = string
}

variable "db_security_group_id" {
  description = "Security group ID for database access"
  type        = string
}
