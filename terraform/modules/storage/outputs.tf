output "training_data_bucket_name" {
  description = "Training data S3 bucket name"
  value       = aws_s3_bucket.training_data.id
}

output "training_data_bucket_arn" {
  description = "Training data S3 bucket ARN"
  value       = aws_s3_bucket.training_data.arn
}

output "backups_bucket_name" {
  description = "Backups S3 bucket name"
  value       = aws_s3_bucket.backups.id
}

output "backups_bucket_arn" {
  description = "Backups S3 bucket ARN"
  value       = aws_s3_bucket.backups.arn
}

output "terraform_state_bucket_name" {
  description = "Terraform state S3 bucket name"
  value       = aws_s3_bucket.terraform_state.id
}

output "terraform_state_bucket_arn" {
  description = "Terraform state S3 bucket ARN"
  value       = aws_s3_bucket.terraform_state.arn
}

output "dynamodb_table_name" {
  description = "DynamoDB table for Terraform state locking"
  value       = aws_dynamodb_table.terraform_locks.name
}
