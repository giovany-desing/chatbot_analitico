# output "alb_dns_name" {
#   description = "ALB DNS name"
#   value       = aws_lb.main.dns_name
# }

# output "alb_arn" {
#   description = "ALB ARN"
#   value       = aws_lb.main.arn
# }

# output "alb_zone_id" {
#   description = "ALB Hosted Zone ID"
#   value       = aws_lb.main.zone_id
# }

# output "target_group_arn" {
#   description = "Target Group ARN"
#   value       = aws_lb_target_group.app.arn
# }

output "instance_ids" {
  description = "EC2 instance IDs"
  value       = aws_instance.app[*].id
}

output "instance_public_ips" {
  description = "EC2 instance public IPs"
  value       = aws_instance.app[*].public_ip
}

output "instance_private_ips" {
  description = "EC2 instance private IPs"
  value       = aws_instance.app[*].private_ip
}

output "ec2_role_arn" {
  description = "EC2 IAM Role ARN"
  value       = aws_iam_role.ec2_role.arn
}

output "ec2_instance_profile_arn" {
  description = "EC2 Instance Profile ARN"
  value       = aws_iam_instance_profile.ec2_profile.arn
}
