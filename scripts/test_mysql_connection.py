#!/usr/bin/env python3
"""
Script para diagnosticar problemas de conexión a MySQL RDS.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import socket
import time
from app.config import settings

def test_dns_resolution(host):
    """Verifica que el DNS resuelva correctamente"""
    try:
        ip = socket.gethostbyname(host)
        print(f"✅ DNS Resolution: {host} -> {ip}")
        return True
    except socket.gaierror as e:
        print(f"❌ DNS Resolution failed: {e}")
        return False

def test_port_connectivity(host, port, timeout=5):
    """Verifica que el puerto esté abierto"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Port {port} is open on {host}")
            return True
        else:
            print(f"❌ Port {port} is closed or unreachable on {host}")
            print(f"   Error code: {result}")
            return False
    except Exception as e:
        print(f"❌ Port connectivity test failed: {e}")
        return False

def test_mysql_connection():
    """Intenta conectar a MySQL"""
    try:
        from app.db.connections import get_mysql
        mysql = get_mysql()
        result = mysql.health_check()
        if result:
            print("✅ MySQL connection successful")
            return True
        else:
            print("❌ MySQL health check failed")
            return False
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        return False

def main():
    print("=" * 60)
    print("MySQL RDS Connection Diagnostic")
    print("=" * 60)
    print()
    
    print(f"Configuration:")
    print(f"  Host: {settings.MYSQL_HOST}")
    print(f"  Port: {settings.MYSQL_PORT}")
    print(f"  Database: {settings.MYSQL_DATABASE}")
    print(f"  User: {settings.MYSQL_USER}")
    print()
    
    # Test 1: DNS Resolution
    print("1. Testing DNS Resolution...")
    dns_ok = test_dns_resolution(settings.MYSQL_HOST)
    print()
    
    if not dns_ok:
        print("❌ Cannot resolve DNS. Check:")
        print("   - RDS endpoint is correct")
        print("   - Network connectivity")
        return
    
    # Test 2: Port Connectivity
    print("2. Testing Port Connectivity...")
    port_ok = test_port_connectivity(settings.MYSQL_HOST, settings.MYSQL_PORT, timeout=10)
    print()
    
    if not port_ok:
        print("❌ Port is not accessible. Check:")
        print("   - RDS Security Group allows inbound connections on port 3306")
        print("   - Security Group source should be:")
        print("     * Your IP address (for local testing)")
        print("     * VPC CIDR block (for VPC connections)")
        print("     * Security Group ID (for cross-SG connections)")
        print("   - Network ACLs allow traffic")
        print("   - RDS instance is in 'available' state")
        return
    
    # Test 3: MySQL Connection
    print("3. Testing MySQL Connection...")
    mysql_ok = test_mysql_connection()
    print()
    
    if mysql_ok:
        print("=" * 60)
        print("✅ All tests passed! MySQL connection is working.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ MySQL connection failed. Check:")
        print("   - Credentials are correct")
        print("   - Database exists")
        print("   - User has proper permissions")
        print("=" * 60)

if __name__ == "__main__":
    main()

