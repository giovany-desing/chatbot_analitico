output "parameter_names" {
  description = "Map of all Parameter Store parameter names"
  value = {
    db_host             = aws_ssm_parameter.db_host.name
    db_name             = aws_ssm_parameter.db_name.name
    db_username         = aws_ssm_parameter.db_username.name
    db_password         = aws_ssm_parameter.db_password.name
    db_port             = aws_ssm_parameter.db_port.name
    groq_api_key        = aws_ssm_parameter.groq_api_key.name
    modal_api_key       = aws_ssm_parameter.modal_api_key.name
    external_mysql_uri  = aws_ssm_parameter.external_mysql_uri.name
    s3_training_bucket  = aws_ssm_parameter.s3_training_bucket.name
    s3_backups_bucket   = aws_ssm_parameter.s3_backups_bucket.name
  }
}

output "parameter_arns" {
  description = "Map of all Parameter Store parameter ARNs"
  value = {
    db_host             = aws_ssm_parameter.db_host.arn
    db_name             = aws_ssm_parameter.db_name.arn
    db_username         = aws_ssm_parameter.db_username.arn
    db_password         = aws_ssm_parameter.db_password.arn
    db_port             = aws_ssm_parameter.db_port.arn
    groq_api_key        = aws_ssm_parameter.groq_api_key.arn
    modal_api_key       = aws_ssm_parameter.modal_api_key.arn
    external_mysql_uri  = aws_ssm_parameter.external_mysql_uri.arn
    s3_training_bucket  = aws_ssm_parameter.s3_training_bucket.arn
    s3_backups_bucket   = aws_ssm_parameter.s3_backups_bucket.arn
  }
}

