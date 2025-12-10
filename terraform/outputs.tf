output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

# output "alb_dns_name" {
#   description = "Application Load Balancer DNS"
#   value       = module.compute.alb_dns_name
# }

# output "alb_url" {
#   description = "Application Load Balancer URL"
#   value       = "http://${module.compute.alb_dns_name}"
# }

output "ec2_access_url" {
  description = "Direct access to EC2 instance (http://PUBLIC_IP:8000)"
  value       = length(module.compute.instance_public_ips) > 0 ? "http://${module.compute.instance_public_ips[0]}:8000" : "No instance available"
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.database.db_endpoint
  sensitive   = true
}

output "s3_training_bucket" {
  description = "S3 bucket for training data"
  value       = module.storage.training_data_bucket_name
}

output "s3_backups_bucket" {
  description = "S3 bucket for backups"
  value       = module.storage.backups_bucket_name
}

output "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  value       = module.storage.terraform_state_bucket_name
}

output "ec2_instance_ids" {
  description = "EC2 instance IDs"
  value       = module.compute.instance_ids
}

output "ec2_public_ips" {
  description = "EC2 instance public IPs"
  value       = module.compute.instance_public_ips
}

output "parameter_store_names" {
  description = "Parameter Store parameter names"
  value       = module.secrets.parameter_names
}

output "cloudwatch_log_group" {
  description = "CloudWatch Log Group para logs de la aplicación"
  value       = module.monitoring.log_group_name
}

output "sns_topic_arn" {
  description = "SNS Topic ARN para alarmas"
  value       = module.monitoring.sns_topic_arn
}

output "dashboard_url" {
  description = "URL del CloudWatch Dashboard"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${module.monitoring.dashboard_name}"
}

output "budget_name" {
  description = "Nombre del presupuesto AWS configurado"
  value       = module.billing.budget_name
}

