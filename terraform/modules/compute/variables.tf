variable "project_name" {
  description = "Project name"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs for ALB and EC2"
  type        = list(string)
}

variable "alb_security_group_id" {
  description = "Security group ID for ALB"
  type        = string
}

variable "app_security_group_id" {
  description = "Security group ID for app instances"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"  # Free Tier eligible
}

variable "instance_count" {
  description = "Number of EC2 instances"
  type        = number
  default     = 1  # Empezar con 1, escalar a 2 para HA
}

variable "ssh_key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "s3_training_bucket_arn" {
  description = "S3 training bucket ARN"
  type        = string
}

variable "s3_backups_bucket_arn" {
  description = "S3 backups bucket ARN"
  type        = string
}
