#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "=========================================="
echo "Starting EC2 User Data Script"
echo "Project: ${project_name}"
echo "Environment: ${environment}"
echo "Region: ${aws_region}"
echo "=========================================="

# Update system
echo "Updating system packages..."
dnf update -y

# Install Docker
echo "Installing Docker..."
dnf install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install Docker Compose
echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION="2.24.5"
curl -L "https://github.com/docker/compose/releases/download/v$DOCKER_COMPOSE_VERSION/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

# Install AWS CLI v2
echo "Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
dnf install -y unzip
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Install Git
echo "Installing Git..."
dnf install -y git

# Create app directory
echo "Creating application directory..."
mkdir -p /opt/chatbot
chown ec2-user:ec2-user /opt/chatbot

# Create systemd service for the app
echo "Creating systemd service..."
cat > /etc/systemd/system/chatbot.service <<'EOF'
[Unit]
Description=Chatbot Analitico
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/chatbot
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ec2-user

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

echo "=========================================="
echo "User Data Script Completed"
echo "=========================================="
