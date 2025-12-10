variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "chatbot-analitico"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "chatbot_rag"
}

variable "db_username" {
  description = "PostgreSQL master username"
  type        = string
  default     = "postgres"
}

variable "db_password" {
  description = "PostgreSQL master password"
  type        = string
  sensitive   = true
}

variable "ssh_key_name" {
  description = "SSH key pair name for EC2 access"
  type        = string
}

variable "groq_api_key" {
  description = "Groq API Key"
  type        = string
  sensitive   = true
  default     = "PLACEHOLDER"
}

variable "modal_api_key" {
  description = "Modal API Key"
  type        = string
  sensitive   = true
  default     = "PLACEHOLDER"
}

variable "external_mysql_uri" {
  description = "External MySQL connection URI"
  type        = string
  sensitive   = true
  default     = "PLACEHOLDER"
}

variable "monitoring_email" {
  description = "Email para recibir alertas de CloudWatch"
  type        = string
  default     = "tu-email@example.com"
}

variable "monthly_budget_limit" {
  description = "Límite de presupuesto mensual en USD"
  type        = number
  default     = 25
}

