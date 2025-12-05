#!/usr/bin/env python3
"""
Script para obtener la IP pública actual.
Útil para configurar el Security Group de RDS.
"""

import urllib.request
import json

def get_public_ip():
    """Obtiene la IP pública usando servicios externos"""
    services = [
        "https://api.ipify.org?format=json",
        "https://ifconfig.me/ip",
        "https://icanhazip.com"
    ]
    
    for service in services:
        try:
            if "ipify" in service:
                response = urllib.request.urlopen(service, timeout=5)
                data = json.loads(response.read().decode())
                return data.get('ip', 'Unknown')
            else:
                response = urllib.request.urlopen(service, timeout=5)
                ip = response.read().decode().strip()
                if ip:
                    return ip
        except Exception as e:
            continue
    
    return "Could not determine IP"

if __name__ == "__main__":
    print("=" * 60)
    print("Your Public IP Address")
    print("=" * 60)
    print()
    
    ip = get_public_ip()
    print(f"Your public IP: {ip}")
    print()
    print("=" * 60)
    print("Instructions for RDS Security Group:")
    print("=" * 60)
    print()
    print("1. Go to AWS Console → RDS → Databases")
    print("2. Select your RDS instance: textil")
    print("3. Click on 'Connectivity & security' tab")
    print("4. Click on the Security Group (e.g., sg-xxxxx)")
    print("5. Click 'Edit inbound rules'")
    print("6. Click 'Add rule'")
    print("7. Configure:")
    print(f"   - Type: MySQL/Aurora")
    print(f"   - Port: 3306")
    print(f"   - Source: {ip}/32")
    print("8. Click 'Save rules'")
    print()
    print("Note: If running from Docker, you may need to:")
    print("  - Add the Docker host IP")
    print("  - Or configure VPC peering if in AWS")
    print("=" * 60)

