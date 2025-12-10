terraform {
  required_version = ">= 1.6.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "chatbot-analitico-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "chatbot-analitico-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "ChatbotAnalitico"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Modules
module "networking" {
  source = "./modules/networking"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr
}

module "database" {
  source = "./modules/database"

  project_name           = var.project_name
  environment            = var.environment
  db_name                = var.db_name
  db_username            = var.db_username
  db_password            = var.db_password
  db_subnet_group_name   = module.networking.db_subnet_group_name
  db_security_group_id   = module.networking.db_security_group_id
}

module "storage" {
  source = "./modules/storage"

  project_name = var.project_name
  environment  = var.environment
}

module "secrets" {
  source = "./modules/secrets"

  project_name = var.project_name
  environment  = var.environment

  # Database credentials
  db_endpoint  = module.database.db_address
  db_name      = var.db_name
  db_username  = var.db_username
  db_password  = var.db_password

  # S3 buckets
  s3_training_bucket = module.storage.training_data_bucket_name
  s3_backups_bucket  = module.storage.backups_bucket_name
  
  # API keys (se actualizarán manualmente después del deploy)
  groq_api_key       = var.groq_api_key
  modal_api_key      = var.modal_api_key
  external_mysql_uri = var.external_mysql_uri
}

module "compute" {
  source = "./modules/compute"

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region

  # Networking
  vpc_id                 = module.networking.vpc_id
  public_subnet_ids      = module.networking.public_subnet_ids
  alb_security_group_id  = module.networking.alb_security_group_id
  app_security_group_id  = module.networking.app_security_group_id

  # S3 buckets for IAM permissions
  s3_training_bucket_arn = module.storage.training_data_bucket_arn
  s3_backups_bucket_arn  = module.storage.backups_bucket_arn
  
  # EC2 configuration
  instance_type  = "t3.micro"
  instance_count = 1
  ssh_key_name   = var.ssh_key_name
}

