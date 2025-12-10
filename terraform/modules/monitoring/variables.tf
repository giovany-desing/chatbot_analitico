variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "ec2_instance_id" {
  description = "EC2 instance ID for monitoring"
  type        = string
}

variable "rds_instance_id" {
  description = "RDS instance ID for monitoring"
  type        = string
}

variable "sns_email" {
  description = "Email address for SNS notifications"
  type        = string
}

