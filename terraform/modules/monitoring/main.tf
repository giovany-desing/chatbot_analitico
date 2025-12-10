# CloudWatch Log Group para logs de la aplicación
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/aws/ec2/${var.project_name}-${var.environment}/app"
  retention_in_days = 7  # Free Tier: 5GB de logs, 7 días de retención

  tags = {
    Name        = "${var.project_name}-${var.environment}-app-logs"
    Environment = var.environment
  }
}

# CloudWatch Log Stream para EC2 (se crea automáticamente cuando se envían logs)
# No necesitamos crear el stream manualmente, se crea al primer log

