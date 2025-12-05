#!/bin/bash
# Script para probar conectividad a RDS desde el contenedor

echo "Testing RDS connectivity from Docker container..."
echo ""

# Test DNS resolution
echo "1. Testing DNS resolution..."
python3 -c "
import socket
try:
    ip = socket.gethostbyname('textil.cof2oucystyr.us-east-1.rds.amazonaws.com')
    print(f'✅ DNS resolved: {ip}')
except Exception as e:
    print(f'❌ DNS failed: {e}')
"

echo ""
echo "2. Testing port connectivity..."
python3 -c "
import socket
import sys

host = 'textil.cof2oucystyr.us-east-1.rds.amazonaws.com'
port = 3306

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    result = sock.connect_ex((host, port))
    sock.close()
    
    if result == 0:
        print(f'✅ Port {port} is accessible')
        sys.exit(0)
    else:
        print(f'❌ Port {port} is NOT accessible (error code: {result})')
        print('')
        print('Possible causes:')
        print('1. Security Group is blocking (but we saw it allows 0.0.0.0/0)')
        print('2. Docker network routing issue')
        print('3. Firewall on host machine')
        print('4. RDS is in private subnet without NAT Gateway')
        sys.exit(1)
except Exception as e:
    print(f'❌ Connection test failed: {e}')
    sys.exit(1)
"

