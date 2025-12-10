# ============================================
# Application Load Balancer
# ============================================
# NOTA: ALB requiere aprobación de AWS Support para cuentas nuevas
# Mientras tanto, accederemos directamente a la IP pública de EC2

# resource "aws_lb" "main" {
#   name               = "${var.project_name}-${var.environment}-alb"
#   internal           = false
#   load_balancer_type = "application"
#   security_groups    = [var.alb_security_group_id]
#   subnets            = var.public_subnet_ids
#
#   enable_deletion_protection = false
#   enable_http2              = true
#   enable_cross_zone_load_balancing = true
#
#   tags = {
#     Name        = "${var.project_name}-${var.environment}-alb"
#     Environment = var.environment
#   }
# }

# Target Group para las instancias EC2
# resource "aws_lb_target_group" "app" {
#   name     = "${var.project_name}-${var.environment}-tg"
#   port     = 8000
#   protocol = "HTTP"
#   vpc_id   = var.vpc_id
#
#   health_check {
#     enabled             = true
#     path                = "/health"
#     port                = "8000"
#     protocol            = "HTTP"
#     healthy_threshold   = 2
#     unhealthy_threshold = 3
#     timeout             = 5
#     interval            = 30
#     matcher             = "200"
#   }
#
#   deregistration_delay = 30
#
#   tags = {
#     Name        = "${var.project_name}-${var.environment}-tg"
#     Environment = var.environment
#   }
# }

# Listener HTTP (puerto 80)
# resource "aws_lb_listener" "http" {
#   load_balancer_arn = aws_lb.main.arn
#   port              = "80"
#   protocol          = "HTTP"
#
#   default_action {
#     type             = "forward"
#     target_group_arn = aws_lb_target_group.app.arn
#   }
#
#   tags = {
#     Name        = "${var.project_name}-${var.environment}-listener-http"
#     Environment = var.environment
#   }
# }

# ============================================
# IAM Role para EC2 (acceso a Parameter Store y S3)
# ============================================

resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-${var.environment}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "${var.project_name}-${var.environment}-ec2-role"
    Environment = var.environment
  }
}

# Policy para acceso a Parameter Store
resource "aws_iam_role_policy" "parameter_store_access" {
  name = "${var.project_name}-${var.environment}-ssm-policy"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:*:parameter/${var.project_name}/${var.environment}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = "*"
      }
    ]
  })
}

# Policy para acceso a S3
resource "aws_iam_role_policy" "s3_access" {
  name = "${var.project_name}-${var.environment}-s3-policy"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "${var.s3_training_bucket_arn}",
          "${var.s3_training_bucket_arn}/*",
          "${var.s3_backups_bucket_arn}",
          "${var.s3_backups_bucket_arn}/*"
        ]
      }
    ]
  })
}

# Instance Profile
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-${var.environment}-ec2-profile"
  role = aws_iam_role.ec2_role.name

  tags = {
    Name        = "${var.project_name}-${var.environment}-ec2-profile"
    Environment = var.environment
  }
}

# ============================================
# EC2 Instances
# ============================================

# Data source para obtener la AMI más reciente de Amazon Linux 2023
data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Launch Template para las instancias
resource "aws_launch_template" "app" {
  name_prefix   = "${var.project_name}-${var.environment}-"
  image_id      = data.aws_ami.amazon_linux_2023.id
  instance_type = var.instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.ec2_profile.arn
  }

  vpc_security_group_ids = [var.app_security_group_id]

  key_name = var.ssh_key_name

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    project_name = var.project_name
    environment  = var.environment
    aws_region   = var.aws_region
  }))

  # Expandir volumen EBS a 16GB (suficiente para torch + transformers)
  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 16  # GB (suficiente para torch + transformers)
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  monitoring {
    enabled = false  # Detailed monitoring no incluido en Free Tier
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"  # Require IMDSv2
    http_put_response_hop_limit = 1
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${var.project_name}-${var.environment}-app"
      Environment = var.environment
    }
  }

  tag_specifications {
    resource_type = "volume"
    tags = {
      Name        = "${var.project_name}-${var.environment}-volume"
      Environment = var.environment
    }
  }
}

# Instancias EC2 (una por subnet pública para HA)
resource "aws_instance" "app" {
  count = var.instance_count

  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }

  subnet_id = var.public_subnet_ids[count.index % length(var.public_subnet_ids)]

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-app-${count.index + 1}"
    Environment = var.environment
  }
}

# Attachment de instancias al Target Group
# resource "aws_lb_target_group_attachment" "app" {
#   count = var.instance_count
#
#   target_group_arn = aws_lb_target_group.app.arn
#   target_id        = aws_instance.app[count.index].id
#   port             = 8000
# }
